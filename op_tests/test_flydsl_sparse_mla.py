# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL gfx942 absorbed sparse MLA (flydsl_sparse_mla_fwd) vs a torch reference.

Token t of every head h attends only to its selected KV rows::

    out[t, h] = softmax(scale * q[t, h] @ kv[sel_t].T, sink[h]) @ kv[sel_t]

Q is BF16 [T, 16, 512], kv a BF16 [pool, 512] latent cache, and sel_t one row of
int32 slots in one of three index formats: dense [T, K] (-1 is skipped), dense
with topk_length [T], or CSR (flat slots plus indptr [T+1]). The optional
sink is a per-head FP32 logit that takes softmax mass but adds no value.

Suites:
  accuracy     the index formats, ragged and empty rows, paired-tile tails, the
               split-K decode path and both V^T routes (top-k score order and
               ascending rows, with and without ordered_indices). Row 0 of kv
               is NaN so a -1 slot that reads memory fails, and the output sits
               between guard rows that must stay untouched.
  stress       sharp logits, values near 2^100 and duplicate slots stay finite.
  scale        FP32 scale edges, including subnormals and the largest legal scale.
  empty_rows   rows with no valid slot write zeros for every split count.
  empty_allocation
               zero-length index tensors write zeros.
  graph        graph replay with indptr rewritten in place and a poisoned
               workspace.
  routes       the V^T route each launch compiles; order_plan keeps the
               caller's indices.
  contract     invalid scales, geometry, devices and metadata are rejected
               before any compile or launch.
  index_order  the physical-page reorder op on full 16384 x 2048 workloads.
  performance  GLM-5.3-Flash chunked prefill and decode on a 4 GiB paged pool
               (3640 blocks x 1152 rows, one 131072-token request), against
               Triton's sparse_mla_fwd and pa_prefill_sparse (T >= 2048) or
               pa_decode_sparse. Rows come in top-k score order (``topk``),
               ascending logical positions mapped through the block table
               (``logical``), or sorted physical slots (``physical``). Caches
               stay warm; op_benchmarks/triton/bench_sparse_mla.py flushes
               them and times sparse_mla_fwd with and without the FlyDSL route.

Usage:
    python op_tests/test_flydsl_sparse_mla.py
    python op_tests/test_flydsl_sparse_mla.py --suite performance -t 16384 --order topk
"""

import argparse
import itertools
import math
import struct
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import sparse_mla as serving
from aiter.ops.flydsl.sparse_mla import (
    flydsl_sparse_mla_fwd,
    order_sparse_mla_indices,
    prepare_sparse_mla_order,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
H, D = 16, 512
TOPK = 2048
PAGE = 1152
POOL_ROWS = (
    3640 * PAGE
)  # every block of the server's cache; int32 offsets need < 2^31 / D
SEQ_LEN = 131072
TAIL_LENGTHS = (
    0,
    1,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    95,
    96,
    97,
    129,
    145,
    2047,
    2049,
)

# family -> argument lists of test_sparse_mla; each family runs their product.
ACCURACY = {
    # Zero to three two-tile loop trips plus the peeled tail, both V^T routes.
    "tiles": {
        "tokens": [1, 3, 7],
        "width": [32, 64, 96, 128],
        "layout": ["dense"],
        "rows": ["full"],
        "has_sink": [False, True],
        "num_splits": [1],
        "order": ["topk"],
        "ordered_indices": [False, True],
    },
    # Ragged and empty rows; 4097 leaves a partial four-token workgroup.
    "ragged": {
        "tokens": [1, 3, 7, 64, 4097],
        "width": [65],
        "layout": ["dense", "dense_lengths", "csr"],
        "rows": ["ragged"],
        "has_sink": [False, True],
    },
    # Every length around 16-row tiles and paired fetches, split and unsplit.
    "tails": {
        "tokens": [len(TAIL_LENGTHS)],
        "width": [2049],
        "layout": ["dense_lengths", "csr"],
        "rows": ["tails"],
        "has_sink": [False, True],
        "num_splits": [1, 5, 44],
        "ordered_indices": [False, True],
    },
    # Batches below 2048 tokens take split-K plus the combine kernel.
    "decode": {
        "tokens": [1, 7, 32, 64, 256],
        "width": [TOPK],
        "layout": ["dense"],
        "rows": ["full"],
        "has_sink": [False, True],
    },
    # Both V^T routes on rows in score order and in ascending order.
    "routes": {
        "tokens": [5, 4097],
        "width": [32, TOPK],
        "layout": ["dense"],
        "rows": ["full"],
        "has_sink": [False, True],
        "num_splits": [1],
        "order": ["topk", "physical"],
        "ordered_indices": [False, True],
    },
    # max_seqlen only plans CSR splits; it never truncates a row.
    "csr_hint": {
        "tokens": [3],
        "width": [65],
        "layout": ["csr"],
        "rows": ["ragged"],
        "has_sink": [False],
        "max_seqlen": [0, 1, 16, TOPK],
    },
}
INVALID_SCALES = (
    0.0,
    -1.0,
    math.nan,
    math.inf,
    1e-50,
    1e-46,
    2.0**-150,
    2.3586576387363357e38,  # scale * LOG2E overflows FP32
    3e38,
    3.4028234663852886e38,
    1e39,
)
GEOMETRY = ("dtype", "heads", "dim", "out", "strides")
OPERANDS = ("q", "kv", "out", "kv_indices", "kv_indptr", "attn_sink")
METADATA = (
    "index_dtype",
    "index_shape",
    "indptr_shape",
    "length_dtype",
    "length_shape",
    "csr_lengths",
    "sink_dtype",
    "sink_shape",
    "splits_zero",
    "splits_large",
    "workspace",
    "max_seqlen",
)
CONTRACT = (
    [f"scale={scale!r}" for scale in INVALID_SCALES]
    + [f"geometry={case}" for case in GEOMETRY]
    + ["capacity=tokens", "capacity=kv"]
    + [f"device={operand}" for operand in OPERANDS]
    + [f"metadata={case}" for case in METADATA]
    + ["order_plan=lengths", "order_plan=width", "arch=gfx950"]
)


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def run_torch(q, kv, rows, sink=None, scale=D**-0.5, chunk=32):
    """FP32 reference over dense [T, K] slots; -1 slots are skipped and an empty
    row without a sink returns zero."""
    out = torch.empty(q.shape, dtype=dtypes.fp32)
    bias = torch.full((H,), -math.inf) if sink is None else sink.float()
    for first in range(0, q.shape[0], chunk):
        slots = rows[first : first + chunk]
        valid = slots >= 0
        values = kv[slots.clamp(min=0).long()].float()
        values.masked_fill_(~valid[..., None], 0)
        scores = torch.einsum("thd,tkd->thk", q[first : first + chunk].float(), values)
        scores = (scores * scale).masked_fill(~valid[:, None, :], -math.inf)
        scores = torch.cat((scores, bias.view(1, H, 1).expand(len(slots), H, 1)), -1)
        probs = torch.softmax(scores, dim=-1)[..., :-1].nan_to_num()
        out[first : first + chunk] = torch.einsum("thk,tkd->thd", probs, values)
    return out


def check_output(ref, out, msg):
    """Every element within 1e-2, and at least 95% within 2e-3 (BF16 rounding)."""
    out = out.float()
    assert torch.isfinite(out).all(), f"{msg}: non-finite output"
    err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, tol_err_ratio=0, msg=msg)
    tight = checkAllclose(ref, out, rtol=2e-3, atol=2e-3, printLog=False)
    assert err == 0 and tight <= 0.05, f"{msg}: mismatch {err} (2e-3: {tight:.3%})"
    return err


def make_paged_indices(tokens, topk, pool, page, seq_len, requests, generator):
    """One chunked-prefill step over a paged KV cache.

    The pool is pool // page blocks; each request owns ceil(seq_len / page) of
    them at random positions. The chunk's queries split evenly over the
    requests, each taking the last tokens / requests positions of its prompt. A
    query at position p selects topk distinct positions of its causal prefix
    [0, p], ascending, mapped through its request's block table.

    Returns physical slots [T, topk], the block table [requests, blocks], the
    logical positions [T, topk] and each row's request [T], all int32.
    """
    blocks = pool // page
    request_blocks = -(-seq_len // page)
    if requests * request_blocks > blocks:
        raise ValueError(f"{requests} requests exceed the {blocks}-block pool")
    if tokens % requests:
        raise ValueError(f"T={tokens} must split evenly over {requests} requests")
    per_request = tokens // requests
    if seq_len - per_request + 1 < topk:
        raise ValueError(f"prefix {seq_len - per_request + 1} < topk {topk}")
    table = torch.randperm(blocks, device="cuda", generator=generator)
    table = table[: requests * request_blocks].view(requests, -1)
    position = torch.arange(tokens)
    prefix = (seq_len - per_request + position % per_request + 1).unsqueeze(1)
    draw = torch.rand(tokens, topk, device="cuda", generator=generator)
    logical = (draw * (prefix - topk + 1)).long().sort(dim=1).values
    logical += torch.arange(topk)
    request = position // per_request
    physical = table[request].gather(1, logical // page) * page + logical % page
    return (
        physical.to(dtypes.i32),
        table.to(dtypes.i32),
        logical.to(dtypes.i32),
        request.to(dtypes.i32),
    )


def make_rows(tokens, width, rows, order, generator, pool=4096):
    """Slots in [1, pool) with interior -1 sentinels, and per-row lengths."""
    if rows == "tails":
        lengths = torch.tensor(TAIL_LENGTHS, dtype=dtypes.i32)
        lengths = lengths[torch.arange(tokens) % len(TAIL_LENGTHS)]
    elif rows == "ragged":
        # T=1 also gets an empty row; all sizes include non-aligned tails.
        lengths = (torch.arange(tokens, dtype=dtypes.i32) * 17) % (width + 1)
    else:
        lengths = torch.full((tokens,), width, dtype=dtypes.i32)
    slots = torch.randint(
        1, pool, (tokens, width), device="cuda", generator=generator, dtype=dtypes.i32
    )
    if order == "physical":
        slots = slots.sort(dim=1).values
    slots[:, 7::23] = -1  # interior sentinels must not end a row
    return slots, lengths


@benchmark()
def test_sparse_mla(
    tokens,
    width,
    layout,
    rows,
    has_sink,
    num_splits=None,
    order="topk",
    ordered_indices=False,
    max_seqlen=None,
):
    generator = torch.Generator(device="cuda").manual_seed(tokens * 7919 + width)
    q = torch.randn((tokens, H, D), device="cuda", generator=generator).to(dtypes.bf16)
    kv = torch.randn((4096, D), device="cuda", generator=generator).to(dtypes.bf16)
    kv[0] = math.nan
    sink = torch.randn((H,), generator=generator, device="cuda") if has_sink else None
    slots, lengths = make_rows(tokens, width, rows, order, generator)
    position = torch.arange(width)
    kept = position[None, :] < lengths[:, None]
    expected = slots.masked_fill(~kept, -1)
    kwargs = {}
    if layout == "csr":
        # Unaligned starts; adjacent rows must not share a paired fetch.
        pad = torch.full((13,), 17, dtype=dtypes.i32)
        indices = torch.cat((pad, slots[kept], pad))
        indptr = torch.zeros(tokens + 1, dtype=dtypes.i32)
        indptr[1:] = lengths.cumsum(0)
        kwargs["kv_indptr"] = indptr + 13
        kwargs["max_seqlen"] = width if max_seqlen is None else max_seqlen
    elif layout == "dense_lengths":
        indices = slots  # positive slots past each length must be ignored
        kwargs["topk_length"] = lengths
    else:
        indices = expected
    guarded = torch.full((tokens + 2, H, D), 42.0, dtype=dtypes.bf16)
    out = guarded[1:-1]
    result = flydsl_sparse_mla_fwd(
        q,
        kv,
        indices,
        out,
        attn_sink=sink,
        num_splits=num_splits,
        ordered_indices=ordered_indices,
        **kwargs,
    )
    assert result is out, "out must be returned"
    assert (guarded[0] == 42).all() and (guarded[-1] == 42).all(), "guard rows changed"
    assert torch.count_nonzero(out[lengths == 0]) == 0, "empty rows must be zero"
    err = check_output(run_torch(q, kv, expected, sink), out, f"{layout} T={tokens}")
    return {"valid slots": int((expected >= 0).sum().item()), "err": err}


@benchmark()
def test_stress(case, tokens):
    """Large and sharp logits, huge values and duplicate slots stay finite."""
    torch.manual_seed(1215)
    q = torch.randn((tokens, H, D), dtype=dtypes.bf16)
    kv = torch.randn((8192, D), dtype=dtypes.bf16)
    rows = torch.arange(TOPK, dtype=dtypes.i32).repeat(tokens, 1)
    sink = None
    if case == "sharp":
        q.mul_(16)
    elif case == "large_v":
        # A stale probability near 2^60 times V = 2^100 overflows FP32; the
        # lazy rescale threshold must keep this finite BF16 case safe.
        q.zero_()
        q[..., 0] = 1
        kv.zero_()
        kv[:, 1] = 2.0**100
        kv[16:TOPK, 0] = 944
    else:
        rows %= 256
        sink = torch.linspace(-2, 2, H)
    out = flydsl_sparse_mla_fwd(q, kv, rows, attn_sink=sink)
    ref = run_torch(q, kv, rows, sink)
    assert torch.isfinite(ref).all()
    return {"err": check_output(ref, out, f"{case} T={tokens}")}


@benchmark()
def test_scale_edge(softmax_scale, has_sink):
    """Representable FP32 scales reach the kernel unchanged and stay finite."""
    q = torch.zeros((1, H, D), dtype=dtypes.bf16)
    kv = torch.randn((256, D), dtype=dtypes.bf16)
    rows = torch.randint(256, (1, 32), dtype=dtypes.i32)
    sink = torch.linspace(-2, 2, H) if has_sink else None
    checked = fp32(softmax_scale)
    assert serving._checked_scale(softmax_scale) == checked
    launch = serving._run_compiled
    scales = []

    def record(*args):
        scales.append(args[7])
        return launch(*args)

    with mock.patch.object(serving, "_run_compiled", record):
        out = flydsl_sparse_mla_fwd(
            q, kv, rows, softmax_scale=softmax_scale, attn_sink=sink, num_splits=1
        )
    assert scales == [checked], f"launched with {scales}, expected {checked}"
    ref = run_torch(q, kv, rows, sink, checked)
    return {"err": check_output(ref, out, f"scale={softmax_scale!r}")}


@benchmark()
def test_empty_rows(num_splits, has_sink):
    """Empty splits and an empty row whose -1 slots would read a NaN row."""
    torch.manual_seed(61)
    q = torch.randn((3, H, D), dtype=dtypes.bf16)
    kv = torch.randn((64, D), dtype=dtypes.bf16)
    kv[0] = math.nan
    rows = torch.full((3, 33), -1, dtype=dtypes.i32)
    rows[1, 0] = 3
    rows[2, :17] = torch.arange(1, 18, dtype=dtypes.i32)
    sink = torch.full((H,), 12.0) if has_sink else None
    out = flydsl_sparse_mla_fwd(q, kv, rows, attn_sink=sink, num_splits=num_splits)
    assert torch.count_nonzero(out[0]) == 0, "empty row must be zero"
    return {"err": check_output(run_torch(q, kv, rows, sink), out, "empty rows")}


@benchmark()
def test_empty_allocation(layout, num_splits):
    """Zero-length index tensors write zeros, even with a -inf sink."""
    q = torch.ones((3, H, D), dtype=dtypes.bf16)
    kv = torch.ones((1, D), dtype=dtypes.bf16)
    shape = (0,) if layout == "csr" else (3, 0)
    indices = torch.empty(shape, dtype=dtypes.i32)
    indptr = torch.zeros((4,), dtype=dtypes.i32) if layout == "csr" else None
    sink = torch.full((H,), -math.inf)
    out = flydsl_sparse_mla_fwd(
        q, kv, indices, kv_indptr=indptr, attn_sink=sink, num_splits=num_splits
    )
    assert torch.count_nonzero(out) == 0, "empty allocation must write zeros"
    return {"nonzero": 0}


@benchmark()
def test_graph_replay(num_splits):
    """One capture, replayed after indptr and the workspace change in place."""
    torch.manual_seed(71)
    q = torch.randn((7, H, D), dtype=dtypes.bf16)
    kv = torch.randn((128, D), dtype=dtypes.bf16)
    indices = torch.randint(128, (7 * 65,), dtype=dtypes.i32)
    indptr = torch.arange(8, dtype=dtypes.i32) * 65
    sink = torch.randn((H,))
    out = torch.empty_like(q)
    workspace = None
    if num_splits > 1:
        workspace = (
            torch.empty((7, num_splits, H, D), dtype=dtypes.fp32),
            torch.empty((7, num_splits, H, 2), dtype=dtypes.fp32),
        )

    def run():
        return flydsl_sparse_mla_fwd(
            q,
            kv,
            indices,
            out,
            kv_indptr=indptr,
            max_seqlen=65,
            attn_sink=sink,
            num_splits=num_splits,
            workspace=workspace,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    errors = [check_output(run_torch(q, kv, indices.view(7, 65), sink), out, "replay")]
    indptr.zero_()
    graph.replay()
    assert torch.count_nonzero(out) == 0, "empty rows after indptr.zero_()"
    if workspace is not None:
        for value in workspace:
            value.fill_(math.nan)
    indptr.copy_(torch.arange(8, dtype=dtypes.i32) * 17)
    graph.replay()
    ref = run_torch(q, kv, indices[: 7 * 17].view(7, 17), sink)
    errors.append(check_output(ref, out, "replay with new indptr"))
    return {"err": max(errors)}


@benchmark()
def test_routes(source):
    """Ordering selects the register V^T route of unsplit launches only, and
    order_plan leaves the caller's indices unchanged."""
    torch.manual_seed(111)
    q = torch.randn((5, H, D), dtype=dtypes.bf16)
    kv = torch.randn((4096, D), dtype=dtypes.bf16)
    kv[0] = math.nan
    rows = torch.randint(1, 4096, (5, TOPK), dtype=dtypes.i32)
    rows[:, 3::29] = -1
    saved = rows.clone()
    kwargs = {}
    if source == "ordered_indices":
        kwargs["ordered_indices"] = True
    elif source == "order_plan":
        table = torch.arange(64, dtype=dtypes.i32).view(1, 64)
        kwargs["order_plan"] = prepare_sparse_mla_order(table, 64, 64)
    compile_fwd = serving.compile_sparse_mla_fwd
    routes = []

    def record(*args, **kw):
        routes.append((kw["split_k"], kw["register_vt"]))
        return compile_fwd(*args, **kw)

    ref = run_torch(q, kv, rows)
    errors = []
    with mock.patch.object(serving, "compile_sparse_mla_fwd", record):
        for num_splits in (1, 5):
            out = flydsl_sparse_mla_fwd(q, kv, rows, num_splits=num_splits, **kwargs)
            errors.append(check_output(ref, out, f"{source} splits={num_splits}"))
    expected = [(1, source != "none"), (5, False)]
    assert routes == expected, f"{source}: routes {routes}, expected {expected}"
    assert torch.equal(rows, saved), f"{source}: caller indices changed"
    return {"routes": routes, "err": max(errors)}


def contract_call(case):
    """Arguments that must be rejected, and the expected error and message."""
    kind, _, value = case.partition("=")
    q = torch.ones((3, H, D), dtype=dtypes.bf16)
    kv = torch.ones((64, D), dtype=dtypes.bf16)
    indices = torch.zeros((3, 32), dtype=dtypes.i32)
    out, kwargs = None, {}
    if kind == "scale":
        kwargs["softmax_scale"] = float(value)
        return (q, kv, indices), kwargs, ValueError, "finite and positive in FP32"
    if kind == "geometry":
        if value == "dtype":
            q = q.float()
        elif value == "heads":
            q = q[:, :8].contiguous()
        elif value == "dim":
            q = q[..., :256].contiguous()
        elif value == "strides":
            q = q[:1].expand(3, -1, -1)
        out = torch.empty_like(q, dtype=dtypes.fp32 if value == "out" else q.dtype)
        return (q, kv, indices, out), kwargs, (TypeError, ValueError), ""
    if kind == "capacity":
        # Metadata-only tensors reach the offset limits without allocating.
        tokens = serving._MAX_TOKENS if value == "tokens" else 1
        pool = serving._MAX_KV_ROWS if value == "kv" else 32
        q = torch.empty((tokens, H, D), dtype=dtypes.bf16, device="meta")
        kv = torch.empty((pool, D), dtype=dtypes.bf16, device="meta")
        return (q, kv, indices, torch.empty_like(q)), kwargs, ValueError, "int32"
    if kind == "device":
        values = {
            "q": q,
            "kv": kv,
            "out": torch.empty_like(q),
            "kv_indices": indices.flatten(),
            "kv_indptr": torch.arange(4, dtype=dtypes.i32) * 32,
            "attn_sink": torch.zeros(H),
        }
        values[value] = values[value].cpu()
        return (), values, ValueError, "GPU"
    if kind == "order_plan":
        table = torch.arange(64, dtype=dtypes.i32).view(1, 64)
        kwargs["order_plan"] = prepare_sparse_mla_order(table, 64, 64)
        if value == "lengths":
            indices = torch.zeros((3, TOPK), dtype=dtypes.i32)
            kwargs["topk_length"] = torch.full((3,), TOPK, dtype=dtypes.i32)
        return (q, kv, indices), kwargs, ValueError, "ordering requires"
    if kind == "arch":
        return (q, kv, indices), kwargs, RuntimeError, "gfx942-only"
    if value == "index_dtype":
        indices = indices.long()
    elif value == "index_shape":
        indices = indices.flatten()
    elif value == "indptr_shape":
        indices = indices.flatten()
        kwargs["kv_indptr"] = torch.zeros(3, dtype=dtypes.i32)
    elif value == "length_dtype":
        kwargs["topk_length"] = torch.zeros(3, dtype=torch.int64)
    elif value == "length_shape":
        kwargs["topk_length"] = torch.zeros(4, dtype=dtypes.i32)
    elif value == "csr_lengths":
        indices = indices.flatten()
        kwargs["kv_indptr"] = torch.arange(4, dtype=dtypes.i32) * 32
        kwargs["topk_length"] = torch.zeros(3, dtype=dtypes.i32)
    elif value == "sink_dtype":
        kwargs["attn_sink"] = torch.zeros(H, dtype=dtypes.bf16)
    elif value == "sink_shape":
        kwargs["attn_sink"] = torch.zeros(H + 1)
    elif value == "splits_zero":
        kwargs["num_splits"] = 0
    elif value == "splits_large":
        kwargs["num_splits"] = 129
    elif value == "workspace":
        kwargs["num_splits"] = 2
        kwargs["workspace"] = (q, q)
    else:
        kwargs["max_seqlen"] = 0.5
    return (q, kv, indices), kwargs, ValueError, ""


@benchmark()
def test_contract(case):
    """Invalid calls raise before any compile or launch."""
    args, kwargs, error, message = contract_call(case)

    def forbidden(*args, **kwargs):
        raise AssertionError(f"{case}: an invalid call reached compile or launch")

    gfx = "gfx950" if case.startswith("arch=") else serving.get_gfx_runtime()
    with mock.patch.object(
        serving, "compile_sparse_mla_fwd", forbidden
    ), mock.patch.object(serving, "_run_compiled", forbidden), mock.patch.object(
        serving, "get_gfx_runtime", lambda: gfx
    ):
        try:
            flydsl_sparse_mla_fwd(*args, **kwargs)
        except error as exc:
            assert message in str(exc), f"{case}: unexpected message {exc}"
            return {"rejected": type(exc).__name__}
    raise AssertionError(f"{case}: invalid call accepted")


@benchmark()
def test_index_order(requests):
    """The reorder op on full workloads: exact multisets, ascending 16-row
    tiles, fused logical mapping, in-place and stream use, and rejects."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    physical, table, logical_rows, req = make_paged_indices(
        16384, TOPK, POOL_ROWS, PAGE, SEQ_LEN, requests, generator
    )
    shuffle = torch.rand(physical.shape, device="cuda", generator=generator).argsort(1)
    physical = physical.gather(1, shuffle)
    logical_rows = logical_rows.gather(1, shuffle)
    plan = prepare_sparse_mla_order(table, PAGE, POOL_ROWS // PAGE)
    order = order_sparse_mla_indices
    checks = 0

    def assert_order(actual, expected):
        nonlocal checks
        same = torch.equal(actual.sort(dim=1).values, expected.sort(dim=1).values)
        assert same, "multiset mismatch"
        tiles = actual // 16
        assert bool((tiles[:, 1:] >= tiles[:, :-1]).all()), "16-row tile order"
        checks += 1

    for logical in (False, True):
        source = logical_rows if logical else physical
        out = torch.empty_like(source)
        result = order(source, plan, out, req, logical=logical)
        assert result.data_ptr() == out.data_ptr()
        assert_order(result, physical)
        inplace = source.clone()
        order(inplace, plan, inplace, req, logical=logical)
        assert_order(inplace, physical)

        # Byte counts 255/256, repeated slots, -1 padding and out-of-plan slots.
        mutated = source.clone()
        mutated[0] = source[0, 0]
        mutated[1, :255] = source[1, 0]
        mutated[2, :256] = source[2, 0]
        mutated[3, :33] = -1
        mutated[4] = -1
        mutated[5, :16] = source[5, 0]
        mutated[6, 0] = 2147483647
        mutated[6, 1] = -2147483648
        for byte in range(4):
            row = 8 + byte
            base = 0 if logical else table[req[row].long(), 0] * PAGE
            mutated[row] = base + byte * 16
        if logical:
            valid = (mutated >= 0) & (mutated // PAGE < table.shape[1])
            safe = torch.where(valid, mutated // PAGE, 0).long()
            mapped = table[req.long()].gather(1, safe) * PAGE + mutated % PAGE
            expected = torch.where(valid, mapped, -1)
        else:
            absent = (plan.physical_rank[req[7].long()] < 0).nonzero()[0, 0]
            mutated[7, :8] = absent.to(dtypes.i32) * PAGE + torch.arange(
                8, dtype=dtypes.i32
            )
            expected = mutated
        order(mutated, plan, out, req, logical=logical)
        assert_order(out, expected)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            order(source, plan, out, req, logical=logical)
        torch.cuda.current_stream().wait_stream(stream)
        assert_order(out, physical)

    if requests == 1:
        # One request needs no row map and no per-call allocation.
        assert_order(order(physical, plan), physical)
        # A contiguous view need not have an aligned storage offset.
        storage = torch.empty(physical.numel() + 1, dtype=dtypes.i32)
        misaligned = storage[1:].view_as(physical)
        misaligned.copy_(physical)
        assert_order(order(misaligned, plan), physical)
        assert order(physical[:0], plan).shape == (0, TOPK)
        graph_out = torch.empty_like(physical)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            order(physical, plan, graph_out, req)
        graph.replay()
        assert_order(graph_out, physical)
        pair = torch.empty(2 * physical.numel(), dtype=dtypes.i32)
        first = pair[: physical.numel()].view_as(physical)
        second = pair[physical.numel() :].view_as(physical)
        first.copy_(physical)
        order(first, plan, second, req)
        assert_order(second, physical)

    overlap = torch.empty(physical.numel() + 1, dtype=dtypes.i32)
    rejects = [
        lambda: order(physical.reshape(-1), plan),
        lambda: order(physical.to(torch.int64), plan),
        lambda: order(physical, plan, out=physical.t()),
        lambda: order(physical, plan, row_requests=req.long()),
        lambda: order(
            overlap[:-1].view_as(physical), plan, overlap[1:].view_as(physical), req
        ),
        lambda: prepare_sparse_mla_order(table, PAGE - 1, POOL_ROWS // PAGE),
        lambda: prepare_sparse_mla_order(
            table[:, :1].expand(-1, 2), PAGE, POOL_ROWS // PAGE
        ),
        lambda: order(physical.t().contiguous().t(), plan, row_requests=req),
    ]
    if requests == 2:
        rejects.append(lambda: order(physical, plan))  # several requests need a map
    for index, call in enumerate(rejects):
        try:
            call()
        except (ValueError, RuntimeError):
            checks += 1
            continue
        raise AssertionError(f"ordering contract {index} accepted")
    return {"checks": checks}


@benchmark()
def test_performance(tokens, order, has_sink):
    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
    from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((tokens, H, D), device="cuda", generator=generator).to(dtypes.bf16)
    kv = torch.randn((POOL_ROWS, D), device="cuda", generator=generator)
    kv = kv.to(dtypes.bf16)
    sink = torch.randn((H,), device="cuda", generator=generator) if has_sink else None
    physical, table, _, _ = make_paged_indices(
        tokens, TOPK, POOL_ROWS, PAGE, SEQ_LEN, 1, generator
    )
    if order == "topk":
        shuffle = torch.rand(physical.shape, device="cuda", generator=generator)
        rows = physical.gather(1, shuffle.argsort(dim=1))
    elif order == "physical":
        rows = physical.sort(dim=1).values
    else:
        rows = physical
    flat = rows.flatten()
    indptr = torch.arange(0, (tokens + 1) * TOPK, TOPK, dtype=dtypes.i32)
    scale = D**-0.5
    plan = prepare_sparse_mla_order(table, PAGE, POOL_ROWS // PAGE)

    # flydsl_ordered on topk rows shows what a wrong ordered_indices hint costs.
    candidates = {
        "flydsl": (flydsl_sparse_mla_fwd, (q, kv, rows), {"attn_sink": sink}),
        "flydsl_ordered": (
            flydsl_sparse_mla_fwd,
            (q, kv, rows),
            {"attn_sink": sink, "ordered_indices": True},
        ),
        "flydsl_order_plan": (
            flydsl_sparse_mla_fwd,
            (q, kv, rows),
            {"attn_sink": sink, "order_plan": plan},
        ),
    }
    if tokens >= 2048:
        candidates["pa_sparse"] = (
            pa_prefill_sparse,
            (q, kv, flat, indptr, None, None, None, sink, scale),
            {},
        )
    else:
        # pa_decode_sparse always applies a sink; exp(-inf) adds nothing.
        candidates["pa_sparse"] = (
            pa_decode_sparse,
            (
                q,
                kv,
                flat,
                indptr,
                torch.full((H,), -math.inf) if sink is None else sink,
            ),
            {"softmax_scale": scale, "has_invalid": False},
        )
    candidates["sparse_mla_fwd"] = (
        lambda *args, **kwargs: sparse_mla_fwd(*args, **kwargs)[0],
        (q, kv, indptr, flat, scale),
        {"qk_rope_head_dim": 0, "attn_sink": sink},
    )

    checked = torch.arange(tokens)
    if tokens > 128:
        checked = torch.cat((checked[:64], checked[-64:]))
    ref = run_torch(q[checked], kv, rows[checked], sink)
    nbytes = (tokens * TOPK * D + 2 * tokens * H * D) * 2 + rows.nbytes
    ret = {"gfx": get_gfx(), "distinct MiB": rows.unique().numel() * D * 2 / 2**20}
    for name, (fn, args, kwargs) in candidates.items():
        out, us = run_perftest(
            fn, *args, num_iters=21, num_warmup=2, num_rotate_args=1, **kwargs
        )
        err = checkAllclose(
            ref,
            out[checked].float(),
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0,
            msg=f"T={tokens} {order} {name}: out",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    # Triton is the baseline only: the FlyDSL candidates must pass the check.
    for name in candidates:
        if name.startswith("flydsl"):
            assert ret[f"{name} err"] == 0, f"T={tokens} {order} {name}: mismatch"
    return ret


def suites(args):
    yield "accuracy", [
        (test_sparse_mla, dict(zip(grid, values)))
        for grid in ACCURACY.values()
        for values in itertools.product(*grid.values())
    ]
    yield "stress", [
        (test_stress, {"case": case, "tokens": tokens})
        for case, tokens in itertools.product(
            ("sharp", "large_v", "duplicates_sink"), (7, 4097)
        )
    ]
    yield "scale", [
        (test_scale_edge, {"softmax_scale": scale, "has_sink": has_sink})
        for scale, has_sink in itertools.product(
            (2.0**-149, 2.0**-126, 0.037, 2.3586574359122396e38), (False, True)
        )
    ]
    yield "empty_rows", [
        (test_empty_rows, {"num_splits": splits, "has_sink": has_sink})
        for splits, has_sink in itertools.product((1, 5, 32, 128), (False, True))
    ]
    yield "empty_allocation", [
        (test_empty_allocation, {"layout": layout, "num_splits": splits})
        for layout, splits in itertools.product(("dense", "csr"), (1, 7))
    ]
    yield "graph", [(test_graph_replay, {"num_splits": n}) for n in (1, 5, 44)]
    yield "routes", [
        (test_routes, {"source": source})
        for source in ("none", "ordered_indices", "order_plan")
    ]
    yield "contract", [(test_contract, {"case": case}) for case in CONTRACT]
    yield "index_order", [(test_index_order, {"requests": n}) for n in (1, 2)]
    yield "performance", [
        (test_performance, {"tokens": t, "order": o, "has_sink": bool(s)})
        for t, o, s in itertools.product(args.tokens, args.order, args.sink)
    ]


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_sparse_mla_fwd unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    names = ("accuracy", "stress", "scale", "empty_rows", "empty_allocation")
    names += ("graph", "routes", "contract", "index_order", "performance")
    parser.add_argument(
        "--suite",
        type=str,
        nargs="*",
        choices=names,
        default=list(names),
        help="suites to run (all by default)",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[64, 16384],
        help="performance: query tokens (below 2048 runs split-K decode)",
    )
    parser.add_argument(
        "--order",
        type=str,
        nargs="*",
        choices=["topk", "logical", "physical"],
        default=["topk", "logical"],
        help="performance: row order (topk = score order, as top-k writes it)",
    )
    parser.add_argument(
        "-s",
        "--sink",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[1],
        help="performance: with (1) or without (0) a per-head attention sink",
    )
    args = parser.parse_args()

    for suite, cases in suites(args):
        if suite not in args.suite:
            continue
        df = [test(**kwargs) for test, kwargs in cases]
        aiter.logger.info(
            "flydsl_sparse_mla_fwd %s summary (markdown):\n%s",
            suite,
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
