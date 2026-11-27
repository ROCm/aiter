# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# QSA regression cases migrated from tingqli/pyhip (MIT), revision 8027faad2.

"""Public QSA indexer correctness with synthetic inputs and independent references."""

import argparse
import inspect
import itertools
import os
import subprocess
import sys
import weakref
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import qsa_indexer as selection
from aiter.ops.flydsl import qsa_indexer_launch as launches
from aiter.ops.triton.attention import qsa_indexer as prep
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests import qsa_indexer_test_utils as helpers

# Sequence/extend lengths cover short rows, the 512-key boundary, packed prefixes, rows crossing 65536 keys
# (int32 chosen ids) and the 1M-token limit.
PREFILL_CASES = (
    ((1,), (1,)),
    ((7,), (7,)),
    ((2051,), (2051,)),
    ((2052,), (2052,)),
    ((2177,), (129,)),
    ((2055, 16, 13), (7, 0, 9)),
    ((262148, 300), (128, 300)),
    ((1048576,), (64,)),
)
# Sequence lengths, padding rows and context: 2044-2052 tokens straddle the 512-key top-k on a 528-key page
# table (not a multiple of 512); 262144 tokens fill the uint16 top-k, longer rows select with int32 ids up to
# the 1M-token limit.
DECODE_CASES = (
    ((1,), 0, 64),
    ((5, 8, 3, 12), 1, 256),
    ((2044, 2048, 2052), 0, 2112),
    ((12000, 11888, 11667, 11851), 0, 16384),
    ((262144,), 0, 262144),
    ((262148, 300000, 9), 1, 300032),
    ((1048576,), 0, 1048576),
)
# TARGET_VERIFY windows: first-row lengths cover every alignment of the compression boundary.
VERIFY_CASES = (
    ((9, 10, 11, 12), 4, 2, 64),
    ((4003, 6, 1023, 2), 4, 1, 8192),
    ((17, 30, 5), 3, 0, 64),
    ((12000, 11889), 2, 3, 16384),
)

PREFILL_CAPTURE_CASES = (
    *[(seq, ext, False) for seq, ext in PREFILL_CASES],
    ((12353,), (65,), True),
)

DECODE_WIDE_GRAPH_CASES = (
    ((262144,), 1, 262144),
    ((262148, 300000, 9), 1, 300032),
    ((1048576,), 1, 1048576),
    ((1048573,), 4, 1048576),
)


@pytest.fixture
def device():
    return helpers._gpu()


def _addresses(source):
    return {
        name: tensor.data_ptr()
        for name, tensor in source.inputs.items()
        if isinstance(tensor, torch.Tensor)
    }


def test_original_public_signatures():
    """SGLang's original keyword dictionaries need no field-name adapter."""
    from aiter.ops.qsa import qsa_decode_indexer, qsa_prefill_indexer

    prefill = [
        "qk",
        "heads",
        "positions",
        "logical_positions",
        "state_slots",
        "key_state",
        "rope_state",
        "write_locs",
        "member_rows",
        "group_sequences",
        "group_ends",
        "rope_matrix",
        "compressed",
        "token_slot_table",
        "cos_sin_cache",
        "axis_map",
        "q_weight",
        "k_weight",
        "q_eps",
        "k_eps",
        "seq_lens",
        "extend_lens",
        "q_out",
        "row_range",
        "out",
    ]
    decode = [
        "qk",
        "positions",
        "logical_positions",
        "state_slots",
        "key_state",
        "rope_state",
        "write_locs",
        "group_locs",
        "compressed",
        "page_table",
        "lengths",
        "cos_sin_cache",
        "axis_map",
        "q_weight",
        "k_weight",
        "q_eps",
        "k_eps",
        "seq_lens",
        "verify",
        "q_out",
        "logits_out",
    ]
    for op, names, defaults in (
        (qsa_prefill_indexer, prefill, {"q_out": None, "row_range": None, "out": None}),
        (
            qsa_decode_indexer,
            decode,
            {"verify": False, "q_out": None, "logits_out": None},
        ),
    ):
        params = inspect.signature(op).parameters
        assert list(params) == names
        for name, param in params.items():
            assert param.kind == (
                inspect.Parameter.POSITIONAL_OR_KEYWORD
                if name == "qk"
                else inspect.Parameter.KEYWORD_ONLY
            )
            assert param.default == defaults.get(name, inspect.Parameter.empty)


PREFILL_PLAN_ERRORS = (
    "heads",
    "group_sequences_dtype",
    "group_ends_shape",
    "group_sequences_cpu",
    "group_ends_cpu",
    "group_sequences_alias",
    "group_ends_alias",
)


@pytest.mark.parametrize("case", PREFILL_PLAN_ERRORS)
def test_prefill_write_plan_validation(case, device):
    source = helpers.synthetic((2177,), (129,), device)
    inputs = dict(source.inputs)
    before = helpers.clone_state(source.state)
    if case == "heads":
        inputs["heads"] = 8
    elif case == "group_sequences_dtype":
        inputs["group_sequences"] = inputs["group_sequences"].int()
    elif case == "group_ends_shape":
        inputs["group_ends"] = inputs["group_ends"][:-1]
    elif case.endswith("_cpu"):
        name = case.removesuffix("_cpu")
        inputs[name] = torch.empty(inputs[name].shape, dtype=torch.int64, device="cpu")
    else:
        name = case.removesuffix("_alias")
        inputs[name] = inputs["rope_state"].view(-1)[: inputs[name].numel()]
    with pytest.raises(ValueError):
        helpers.run_prefill(**inputs)
    for name in helpers.STATE_NAMES:
        assert torch.equal(inputs[name], before[name])


def _check_prefill(source):
    expected = helpers.reference_prep(source.inputs, source.state)
    q = torch.empty_like(expected.q)
    output = helpers.run_prefill(**source.inputs, q_out=q)
    helpers.check(source, actual=output, q=q, expected=expected)
    return output


@pytest.mark.parametrize(
    "seq_lens,extend_lens",
    PREFILL_CASES,
    ids=(
        "short",
        "tail",
        "topk-limit",
        "topk-over",
        "prefix",
        "multi",
        "wide",
        "max-keys",
    ),
)
def test_prefill_operator(seq_lens, extend_lens, device):
    _check_prefill(helpers.synthetic(seq_lens, extend_lens, device))


def test_prefill_logits_chunks(device, monkeypatch):
    """Rows split over many logits chunks (some starting inside a request, some spanning two, all reusing one
    logits buffer) select exactly like the reference, every row checked."""
    monkeypatch.setattr(selection, "_LOGITS_BUDGET_BYTES", 1 << 20)
    monkeypatch.setattr(selection, "_CHUNK_ROWS", 128)
    monkeypatch.setattr(selection, "_layouts", OrderedDict())
    source = helpers.synthetic((10240, 300, 4100), (2048, 300, 4100), device)
    assert (
        len(selection._layout(source.seq_lens, source.extend_lens, device).chunks) > 1
    )
    _check_prefill(source)


@pytest.mark.parametrize(
    "seq_lens,extend_lens",
    (
        ((2055, 16, 13), (7, 0, 9)),
        ((2177, 4100), (129, 4100)),
        ((10240, 300, 4100), (2048, 300, 4100)),
    ),
    ids=("multi", "prefix", "chunks"),
)
def test_prefill_row_ranges(seq_lens, extend_lens, device, monkeypatch):
    """Row ranges (a tensor-parallel rank's share, cutting requests, tiles and logits chunks; possibly empty)
    write exactly the full call's rows into a shared output, leave its other rows alone and still update the
    caches for every row."""
    monkeypatch.setattr(selection, "_LOGITS_BUDGET_BYTES", 1 << 20)
    monkeypatch.setattr(selection, "_CHUNK_ROWS", 128)
    monkeypatch.setattr(selection, "_layouts", OrderedDict())
    source = helpers.synthetic(seq_lens, extend_lens, device)
    expected = helpers.reference_prep(source.inputs, source.state)
    full = helpers.run_prefill(**source.inputs)
    rows = full.shape[0]
    for parts in (2, 3, 7):
        share = -(-rows // parts)
        out = torch.full(
            (share * parts, full.shape[1]), -7, dtype=torch.int32, device=device
        )
        for part in range(parts):
            helpers.reset_state(source)
            first, end = min(part * share, rows), min((part + 1) * share, rows)
            assert (
                helpers.run_prefill(**source.inputs, row_range=(first, end), out=out)
                is out
            )
            helpers.assert_state(source, expected)
            assert bool((out[end:] == -7).all())
        helpers.assert_exact(out[:rows], full, f"{parts} row ranges")


def test_prefill_frame_and_output_ownership(device):
    source = helpers.synthetic((2055, 13), (7, 9), device, cache_dtype=torch.float32)
    # The public ABI allows a noncontiguous RoPE axis stride, but unit token stride.
    storage = torch.empty((3, source.rows + 5), dtype=torch.int64, device=device)
    storage[:, : source.rows].copy_(source.inputs["positions"])
    source.inputs["positions"] = storage[:, : source.rows]
    output = _check_prefill(source)
    saved = output.clone()
    helpers.reset_state(source)
    source.inputs["qk"].neg_()
    later = _check_prefill(source)
    assert later.data_ptr() != output.data_ptr()
    helpers.assert_exact(
        output, saved, "a later call must not overwrite an earlier result"
    )


def test_prefill_position_mismatch_traps(device):
    """A query position off the host prefix+index layout stops the GPU queue in top-k, which aborts the process."""
    script = (
        "import torch\n"
        "from op_tests import qsa_indexer_test_utils as helpers\n"
        "source = helpers.synthetic((2177,), (129,), helpers._gpu())\n"
        "source.inputs['logical_positions'][5] += 1\n"
        "helpers.run_prefill(**source.inputs)\n"
        "torch.cuda.synchronize()\n"
        "print('no trap')\n"
    )
    # The trap is deliberate: skip ROCr's GPU core dump file.
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=900,
        env=dict(os.environ, HSA_DISABLE_COREDUMP_ON_EXCEPTION="1"),
        check=False,
    )
    assert result.returncode != 0 and "no trap" not in result.stdout
    assert (
        "HSA_STATUS_ERROR_EXCEPTION" in result.stderr
        or "qsa_indexer_topk" in result.stderr
    )


def test_prep_kernels_compile_once(device):
    """Row counts, group counts, position strides and decode/verify batches of any value (1, multiples
    of 16, others) reuse one compiled q_prep, k_compress, ring_compress and decode_prep per mode; a
    first prefill whose rows need no logits still compiles the logits kernel."""
    kernels = (
        prep._indexer_q_prep,
        prep._indexer_k_compress,
        prep._indexer_decode_prep,
        prep._indexer_ring_compress,
    )

    def counts():
        return [len(kernel.device_caches[device.index][0]) for kernel in kernels]

    def verify(lengths):
        source = helpers.decode_case(lengths, device, window=4, context=64)
        helpers.run_decode(**source.inputs, verify=True)

    # The first short prefill must warm the scorer without launching work.
    helpers.run_prefill(**helpers.synthetic((9,), (9,), device).inputs)
    from aiter.ops.flydsl.kernels.qsa import indexer_logits

    assert indexer_logits._launch._topk_by_device[device.index]._cf_by_specialization
    helpers.run_decode(**helpers.decode_case((7,), device, context=64).inputs)
    verify((7,))
    first = counts()
    for seq_lens, extend_lens in (
        ((1,), (1,)),
        ((64,), (64,)),
        ((60,), (60,)),
        ((2177,), (129,)),
        ((4096, 63), (4096, 63)),
    ):
        helpers.run_prefill(**helpers.synthetic(seq_lens, extend_lens, device).inputs)
    for rows in (2, 16, 32):
        lengths = tuple(range(5, 5 + rows))
        helpers.run_decode(**helpers.decode_case(lengths, device, context=64).inputs)
        verify(lengths)
    assert counts() == first


@pytest.mark.parametrize(
    "lengths,padding,context",
    DECODE_CASES,
    ids=(
        "short",
        "mixed-padded",
        "topk-boundary",
        "long-ragged",
        "narrow-limit",
        "wide",
        "max-keys",
    ),
)
def test_decode_operator(lengths, padding, context, device):
    source = helpers.decode_case(lengths, device, padding=padding, context=context)
    helpers.check_decode(source)


MALFORMED_DECODE = {
    "qk": lambda t: t[:0],
    "page_table": lambda t: t[:, :0],
    "lengths": lambda t: t.long(),
    "group_locs": lambda t: t[:, :3],
    "seq_lens": lambda t: t[1:],
    "logical_positions": lambda t: t.float(),
    "positions": lambda t: t.int(),
}


@pytest.mark.parametrize("name", MALFORMED_DECODE)
def test_decode_rejects_malformed_inputs(name, device):
    """Malformed decode inputs raise before any launch writes the caches."""
    source = helpers.decode_case((7, 9), device, context=64)
    state = [source.inputs[key].clone() for key in helpers.STATE_NAMES]
    inputs = dict(source.inputs, **{name: MALFORMED_DECODE[name](source.inputs[name])})
    with pytest.raises(ValueError):
        helpers.run_decode(**inputs)
    for key, before in zip(helpers.STATE_NAMES, state):
        assert torch.equal(source.inputs[key], before)


def test_decode_graph_replay(device):
    lengths = (4003, 6, 1023, 2)
    source = helpers.decode_case(lengths, device, padding=2, context=8192)
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    warm = helpers.run_decode(**source.inputs)
    helpers.check_decode(source, warm, expected=expected)
    torch.cuda.synchronize(device)
    helpers.reset_state(source)
    addresses = _addresses(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = helpers.run_decode(**source.inputs)
    # Advance only independent history before crossing the next compression boundary.
    source.state = expected.state
    helpers.reset_state(source)
    helpers.decode_step(source, tuple(length + 1 for length in lengths))
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    graph.replay()
    helpers.check_decode(source, output, expected=expected)
    assert _addresses(source) == addresses


@pytest.mark.parametrize(
    "lengths,window,padding,context",
    VERIFY_CASES,
    ids=("alignments", "long-padded", "window3", "window2-padded"),
)
def test_verify_operator(lengths, window, padding, context, device):
    source = helpers.decode_case(
        lengths, device, window=window, padding=padding, context=context
    )
    helpers.check_decode(source)


def test_verify_graph_replay(device):
    lengths = (4003, 6, 1023, 2)
    source = helpers.decode_case(lengths, device, window=4, padding=2, context=8192)
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    warm = helpers.run_decode(**source.inputs, verify=True)
    helpers.check_decode(source, warm, expected=expected)
    torch.cuda.synchronize(device)
    helpers.reset_state(source)
    addresses = _addresses(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = helpers.run_decode(**source.inputs, verify=True)
    # The next window starts after each request's accepted tokens and rewrites the rejected rows.
    source.state = expected.state
    helpers.reset_state(source)
    helpers.decode_step(
        source,
        tuple(length + accepted for length, accepted in zip(lengths, (4, 1, 3, 2))),
    )
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    graph.replay()
    helpers.check_decode(source, output, expected=expected)
    assert _addresses(source) == addresses


@pytest.mark.parametrize("split", (False, True), ids=("wave-rows", "split-rows"))
@pytest.mark.parametrize(
    "counts,stride",
    (((20000, 600, 65536, 513), 65536), ((70000, 600, 262144, 513, 65537), 262144)),
    ids=("narrow", "wide"),
)
@pytest.mark.parametrize("pattern", ("random", "ties", "dense-threshold", "nan", "inf"))
def test_decode_topk_matches_row_kernel(
    pattern, counts, stride, split, device, monkeypatch
):
    """The CTA-per-row decode top-k reproduces the prefill row kernel (one wave per row, or a CTA's four
    waves per row) bit for bit, with uint16 (rows of at most 65536 keys) and int32 chosen ids.
    """
    from aiter.ops.flydsl.kernels.qsa import indexer_topk

    monkeypatch.setattr(indexer_topk, "SPLIT_KEYS", 0 if split else 1 << 30)
    generator = torch.Generator(device=device).manual_seed(11)
    logits = torch.randn(
        (len(counts), stride), generator=generator, dtype=torch.float32, device=device
    )
    if pattern == "ties":
        logits.fill_(1.0)
    elif pattern == "dense-threshold":
        logits = (logits * 2).round() / 2
    elif pattern == "nan":
        logits[:, 7::997] = float("nan")
    elif pattern == "inf":
        logits[:, 3::1013] = float("inf")
        logits[:, 5::1009] = float("-inf")
    lengths = torch.tensor(counts, dtype=torch.int32, device=device)
    positions = lengths.long() * 4 + 2
    sequences = (positions + 1).int()
    # Both kernels take each row's (min bits, ~max bits) from the logits kernel; NaN orders above +inf.
    causal = torch.arange(stride, device=device)[None] < lengths[:, None]
    finite = causal & ~logits.isnan()
    high = torch.where(finite, logits, float("-inf")).amax(1).view(torch.int32)
    high = torch.where((causal & logits.isnan()).any(1), 0x7FC00000, high)
    stats = torch.stack(
        (torch.where(finite, logits, float("inf")).amin(1).view(torch.int32), ~high),
        dim=1,
    ).contiguous()
    # Decode holds them in the first of the row's stats slots; the rest stay all ones.
    slots = torch.full(
        (len(counts), indexer_topk.STAT_SLOTS, 2), -1, dtype=torch.int32, device=device
    )
    slots[:, 0] = stats
    decode = torch.full((len(counts), 2051), -7, dtype=torch.int32, device=device)
    launches.decode_topk(logits, lengths, positions, sequences, decode, slots)
    row_info = torch.stack((positions.int(), sequences), dim=1).contiguous()
    rows = torch.full_like(decode, -7)
    # The row kernel reads prefill chunks stored key-block-major.
    group = indexer_topk.GROUP_KEYS
    blocked = (
        logits.view(len(counts), stride // group, group).transpose(0, 1).contiguous()
    )
    launches.prefill_topk(
        blocked, stride, 0, len(counts), group, positions, row_info, rows, stats
    )
    torch.cuda.synchronize(device)
    assert torch.equal(decode, rows)
    # Independent total-order reference, including NaN, infinity and stable ties.
    bits = logits.view(torch.int32).long()
    keys = torch.where(bits < 0, ~bits, bits | (1 << 31)) & 0xFFFFFFFF
    keys.masked_fill_(~causal, -1)
    selected = (
        keys.argsort(dim=1, descending=True, stable=True)[:, :512].sort(dim=1).values
    )
    expected = helpers.expand_reference(selected, positions, sequences)
    assert torch.equal(decode, expected)


def test_topk_output_over_int32_elements(device):
    """A 1M-row prefill's int32 [rows, 2051] output holds over 2**31 - 1 elements, which FlyDSL's int32 tensor
    dims cannot describe flattened: the top-k takes it 2-D, and the last rows land where they belong.
    """
    from aiter.ops.flydsl.kernels.qsa import indexer_topk

    counts, stride, total = (4000, 600, 2048, 513), 4096, 1 << 20
    rows, row0 = len(counts), total - len(counts)
    generator = torch.Generator(device=device).manual_seed(5)
    logits = torch.randn(
        (rows, stride), generator=generator, dtype=torch.float32, device=device
    ).abs()
    lengths = torch.tensor(counts, dtype=torch.int32, device=device)
    positions = lengths.long() * 4 + 2
    row_info = torch.stack((positions.int(), (positions + 1).int()), dim=1).contiguous()
    causal = torch.arange(stride, device=device)[None] < lengths[:, None]
    stats = torch.stack(
        (
            torch.where(causal, logits, float("inf")).amin(1).view(torch.int32),
            ~torch.where(causal, logits, float("-inf")).amax(1).view(torch.int32),
        ),
        dim=1,
    ).contiguous()
    group = indexer_topk.GROUP_KEYS
    blocked = logits.view(rows, stride // group, group).transpose(0, 1).contiguous()
    expected = torch.full((rows, 2051), -7, dtype=torch.int32, device=device)
    launches.prefill_topk(
        blocked, stride, 0, rows, group, positions, row_info, expected, stats
    )

    def last_rows(values):
        frame = torch.zeros(
            (total, *values.shape[1:]), dtype=values.dtype, device=device
        )
        frame[row0:] = values
        return frame

    out = torch.empty((total, 2051), dtype=torch.int32, device=device)
    out[row0 - 1 :] = -7
    launches.prefill_topk(
        blocked,
        stride,
        row0,
        rows,
        group,
        last_rows(positions),
        last_rows(row_info),
        out,
        last_rows(stats),
    )
    torch.cuda.synchronize(device)
    assert out.numel() >= 2**31
    assert torch.equal(out[row0:], expected) and bool((out[row0 - 1] == -7).all())


def _yarn_cache(device):
    from aiter.rotary_embedding import YaRNScalingRotaryEmbedding

    # Keep the real YaRN frequency and magnitude scaling, including positions
    # beyond the original 8192-token window. MRoPE axes remain caller-owned.
    with torch.device(device):
        rotary = YaRNScalingRotaryEmbedding(
            128, 64, 8192, 1000000, True, 4.0, torch.bfloat16
        )
    assert rotary.mscale > 1
    return torch.cat(
        (rotary.cos_cache.view(-1, 32), rotary.sin_cache.view(-1, 32)), dim=1
    ).contiguous()


@pytest.mark.parametrize("mode", ("prefill", "decode", "verify"))
@pytest.mark.parametrize("axes", (1, 3))
def test_yarn_prefill_decode_verify(mode, axes, device):
    cache = _yarn_cache(device)
    if mode == "prefill":
        source = helpers.synthetic((12353,), (65,), device, position_axes=axes)
        source.inputs["cos_sin_cache"] = cache
        _check_prefill(source)
        return
    lengths = (12001, 12002, 12003, 12004)
    source = helpers.decode_case(
        lengths, device, window=4 if mode == "verify" else 1, padding=2, context=16384
    )
    source.inputs["cos_sin_cache"] = cache
    if axes == 1:
        source.inputs["positions"] = source.inputs["positions"][0].clone()
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    q = torch.empty_like(expected.q)
    warm = helpers.run_decode(**source.inputs, verify=mode == "verify", q_out=q)
    helpers.check_decode(source, warm, q=q, expected=expected)
    helpers.reset_state(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = helpers.run_decode(**source.inputs, verify=mode == "verify", q_out=q)
    source.state = expected.state
    helpers.reset_state(source)
    if axes == 1:
        # Refresh into a temporary 3-axis frame, then publish the first axis
        # into the original graph input without changing its address.
        one_axis = source.inputs["positions"]
        source.inputs["positions"] = torch.empty(
            (3, source.rows), dtype=torch.int64, device=device
        )
    helpers.decode_step(
        source,
        tuple(
            n + a
            for n, a in zip(lengths, (1, 1, 1, 1) if mode == "decode" else (4, 1, 3, 2))
        ),
    )
    if axes == 1:
        one_axis.copy_(source.inputs["positions"][0])
        source.inputs["positions"] = one_axis
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    graph.replay()
    helpers.check_decode(source, out, q=q, expected=expected)


def test_prefill_rejects_alias(device):
    source = helpers.synthetic((2177,), (1,), device)
    before = helpers.clone_state(source.state)
    with pytest.raises(ValueError, match="overlap"):
        helpers.run_prefill(
            **source.inputs, out=source.inputs["token_slot_table"][:, :2051]
        )
    q_alias = source.inputs["qk"].view(-1)[:512].view(1, 4, 128)
    with pytest.raises(ValueError, match="overlap"):
        helpers.run_prefill(**source.inputs, q_out=q_alias)
    for name in helpers.STATE_NAMES:
        assert torch.equal(source.inputs[name], before[name])


@pytest.mark.parametrize("row_range", ((3, 100), (0, 0)))
def test_prefill_row_range_without_out(row_range, device):
    """Like the original entry, allocate out and define only the selected rows."""
    source = helpers.synthetic((2177,), (129,), device)
    expected = helpers.reference_prep(source.inputs, source.state)
    full = _check_prefill(source)
    helpers.reset_state(source)
    q = torch.full_like(expected.q, float("nan"))
    out = helpers.run_prefill(**source.inputs, row_range=row_range, q_out=q)
    assert out.shape == (source.rows, 2051) and out.dtype == torch.int32
    assert out.data_ptr() != full.data_ptr()
    first, end = row_range
    helpers.assert_exact(out[first:end], full[first:end], "allocated row range")
    helpers.assert_exact(q, expected.q, "row range still prepares all Q")
    helpers.assert_state(source, expected)


def test_decode_logits_padding_and_alias(device):
    source = helpers.decode_case((7, 9), device, context=64)
    before = helpers.clone_state(source.state)
    logits = torch.empty((source.rows, 16), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="512"):
        helpers.run_decode(**source.inputs, logits_out=logits)
    q_alias = (
        source.inputs["key_state"]
        .view(-1)[: source.rows * 512]
        .view(source.rows, 4, 128)
    )
    with pytest.raises(ValueError, match="overlap"):
        helpers.run_decode(**source.inputs, q_out=q_alias)
    for name in helpers.STATE_NAMES:
        assert torch.equal(source.inputs[name], before[name])


def test_prefill_capture_keeps_plan_after_eviction(device):
    source = helpers.synthetic((2177,), (129,), device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        helpers.run_prefill(**source.inputs)
        helpers.reset_state(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = helpers.run_prefill(**source.inputs)
        plan = selection._layout(source.seq_lens, source.extend_lens, device)
        ref = weakref.ref(plan)
        del plan
    torch.cuda.current_stream(device).wait_stream(stream)
    for i in range(10):
        later = helpers.synthetic((2304 + 4 * i,), (128,), device)
        helpers.run_prefill(**later.inputs)
    assert ref() is not None, "Captured metadata was freed by the eager LRU"
    assert all(plan is not ref() for plan in selection._layouts.values())
    helpers.reset_state(source)
    source.inputs["qk"].neg_()
    expected = helpers.reference_prep(source.inputs, source.state)
    graph.replay()
    helpers.check(source, actual=out, expected=expected)


@pytest.mark.parametrize("heads", (12, 6, 3))
def test_indexer_to_sparse_prefill(heads, device):
    from aiter.ops.qsa import qsa_sparse_prefill

    source = helpers.synthetic((2055, 13), (7, 9), device)
    indices = helpers.run_prefill(**source.inputs)
    q = (
        torch.randn((source.rows, heads, 256), dtype=torch.bfloat16, device=device)
        * 0.1
    )
    k = (
        torch.randn((sum(source.seq_lens), 1, 256), dtype=torch.bfloat16, device=device)
        * 0.1
    )
    v = torch.randn(k.shape, dtype=torch.bfloat16, device=device) * 0.1
    actual = qsa_sparse_prefill(
        q,
        k,
        v,
        indices,
        query_lens=source.extend_lens,
        prefix_lens=tuple(s - e for s, e in zip(source.seq_lens, source.extend_lens)),
    )
    expected = torch.empty(q.shape, dtype=torch.float32, device=device)
    first_row = first_key = 0
    for length, rows in zip(source.seq_lens, source.extend_lens):
        ids = indices[first_row : first_row + rows].long()
        keys = k[first_key + ids.clamp_min(0), 0].float()
        values = v[first_key + ids.clamp_min(0), 0].float()
        scores = (
            torch.einsum("rhd,rkd->rhk", q[first_row : first_row + rows].float(), keys)
            / 16
        )
        scores.masked_fill_(ids[:, None] < 0, -torch.inf)
        expected[first_row : first_row + rows] = torch.einsum(
            "rhk,rkd->rhd", scores.softmax(-1), values
        )
        first_row += rows
        first_key += length
    err = checkAllclose(
        expected,
        actual.float(),
        rtol=0.02,
        atol=0.02,
        msg="QSA indexer to sparse prefill",
    )
    assert err == 0


@pytest.mark.parametrize("seq_lens,extend_lens,yarn", PREFILL_CAPTURE_CASES)
def test_prefill_graph_shapes(seq_lens, extend_lens, yarn, device):
    source = helpers.synthetic(seq_lens, extend_lens, device)
    if yarn:
        source.inputs["cos_sin_cache"] = _yarn_cache(device)
    expected = helpers.reference_prep(source.inputs, source.state)
    q = torch.full_like(expected.q, float("nan"))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        warm = helpers.run_prefill(**source.inputs, q_out=q)
        helpers.check(source, actual=warm, q=q, expected=expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = helpers.run_prefill(**source.inputs, q_out=q)
        helpers.reset_state(source)
        graph.replay()
    torch.cuda.current_stream(device).wait_stream(stream)
    helpers.check(source, actual=out, q=q, expected=expected)


@pytest.mark.parametrize("lengths,window,context", DECODE_WIDE_GRAPH_CASES)
def test_decode_wide_graph(lengths, window, context, device):
    source = helpers.decode_case(
        lengths, device, window=window, padding=1, context=context
    )
    helpers.mask_decode_padding(source)
    expected = helpers.reference_prep(source.inputs, source.state, decode=True)
    q = torch.full_like(expected.q, float("nan"))
    warm = helpers.run_decode(**source.inputs, verify=window > 1, q_out=q)
    helpers.check_decode(source, warm, q=q, expected=expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = helpers.run_decode(**source.inputs, verify=window > 1, q_out=q)
    helpers.reset_state(source)
    graph.replay()
    helpers.check_decode(source, out, q=q, expected=expected)


@pytest.mark.parametrize("window", (1, 2, 3, 4))
@pytest.mark.parametrize("yarn", (False, True), ids=("mrope", "yarn"))
@pytest.mark.parametrize("graph", (False, True), ids=("eager", "graph"))
def test_decode_state_trajectory(window, yarn, graph, device):
    lengths = (12001, 12002, 12003, 12004)
    source = helpers.decode_case(
        lengths, device, window=window, padding=2, context=16384
    )
    helpers.mask_decode_padding(source)
    if yarn:
        source.inputs["cos_sin_cache"] = _yarn_cache(device)
    oracle_state = helpers.clone_state(source.state)
    expected = helpers.reference_prep(source.inputs, oracle_state, decode=True)
    q = torch.full_like(expected.q, float("nan"))
    if graph:
        helpers.run_decode(**source.inputs, verify=window > 1, q_out=q)
        full_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_graph):
            out = helpers.run_decode(**source.inputs, verify=window > 1, q_out=q)
        helpers.reset_state(source)
    addresses = _addresses(source)
    for step in range(4):
        if step:
            helpers.decode_step(source, lengths)
            helpers.mask_decode_padding(source)
        expected = helpers.reference_prep(source.inputs, oracle_state, decode=True)
        if graph:
            full_graph.replay()
        else:
            out = helpers.run_decode(**source.inputs, verify=window > 1, q_out=q)
        helpers.check_decode(source, out, q=q, expected=expected)
        assert _addresses(source) == addresses
        # Actual state is never reset between steps. The oracle advances from
        # its own state; the accepted prefix sets the next window's positions.
        oracle_state = expected.state
        accepted = tuple(1 + (step + r) % window for r in range(source.requests))
        lengths = tuple(n + a for n, a in zip(lengths, accepted))

    # Resume ordinary decode from the same ring after partial acceptance.
    next_source = helpers.decode_case(
        lengths, device, padding=2, context=source.context
    )
    next_source.table = source.table
    next_source.generator = source.generator
    next_source.state = oracle_state
    for name in (
        *helpers.STATE_NAMES,
        "q_weight",
        "k_weight",
        "cos_sin_cache",
        "axis_map",
    ):
        next_source.inputs[name] = source.inputs[name]
    helpers.decode_step(next_source, lengths)
    helpers.mask_decode_padding(next_source)
    expected = helpers.reference_prep(next_source.inputs, oracle_state, decode=True)
    q = torch.empty_like(expected.q)
    out = helpers.run_decode(**next_source.inputs, q_out=q)
    helpers.check_decode(next_source, out, q=q, expected=expected)


@benchmark()
def benchmark_indexer(mode, batch, context, query_length):
    """Time projected QK through indices, with the Torch reference outside timing."""
    device = torch.device("cuda", torch.cuda.current_device())
    if mode == "prefill":
        source = helpers.synthetic((context,) * batch, (query_length,) * batch, device)
    else:
        source = helpers.decode_case(
            (context - query_length + 1,) * batch,
            device,
            window=query_length,
            context=-(-context // 64) * 64,
        )
    expected = helpers.reference_prep(
        source.inputs, source.state, decode=mode != "prefill"
    )
    q = torch.full_like(expected.q, float("nan"))
    if mode == "prefill":
        fn = lambda: helpers.run_prefill(**source.inputs, q_out=q)
    else:
        fn = lambda: helpers.run_decode(
            **source.inputs, verify=query_length > 1, q_out=q
        )
    candidates = {"aiter": fn}
    # Dot-product FLOPs and logical tensor traffic; cache reuse changes HBM traffic.
    pairs = (
        sum((context - query_length + i + 1) // 4 for i in range(query_length)) * batch
    )
    flops = pairs * 2 * 4 * 128
    nbytes = q.numel() * 2 + pairs * (128 * 2 + 4)
    nbytes += source.inputs["qk"].numel() * 2
    ret = {
        "gfx": get_gfx_runtime(),
        "boundary": "qk -> indices",
    }
    for name, call in candidates.items():
        helpers.reset_state(source)
        out, us = run_perftest(call)
        err = checkAllclose(
            expected.q.float(),
            q.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: QSA index Q",
        )
        if mode == "prefill":
            helpers.check(source, actual=out, q=q, expected=expected)
        else:
            helpers.check_decode(source, out, q=q, expected=expected)
        assert err == 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() != "gfx942":
        aiter.logger.warning("QSA indexer requires gfx942; skipping")
        return
    parser = argparse.ArgumentParser(
        description="QSA indexer correctness and GPU performance"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("prefill", "decode", "verify"),
        default=["prefill", "decode", "verify"],
    )
    parser.add_argument("-b", "--batch", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--contexts", nargs="+", type=int, default=[12000])
    args = parser.parse_args()
    device = helpers._gpu()
    test_original_public_signatures()
    for case in PREFILL_PLAN_ERRORS:
        test_prefill_write_plan_validation(case, device)
    for row_range in ((3, 100), (0, 0)):
        test_prefill_row_range_without_out(row_range, device)
    for seq, ext in PREFILL_CASES:
        test_prefill_operator(seq, ext, device)
    for lengths, padding, context in DECODE_CASES:
        test_decode_operator(lengths, padding, context, device)
    for lengths, window, padding, context in VERIFY_CASES:
        test_verify_operator(lengths, window, padding, context, device)
    with pytest.MonkeyPatch.context() as patch:
        test_prefill_logits_chunks(device, patch)
    for seq, ext in (
        ((2055, 16, 13), (7, 0, 9)),
        ((2177, 4100), (129, 4100)),
        ((10240, 300, 4100), (2048, 300, 4100)),
    ):
        with pytest.MonkeyPatch.context() as patch:
            test_prefill_row_ranges(seq, ext, device, patch)
    for test in (
        test_prefill_frame_and_output_ownership,
        test_prefill_position_mismatch_traps,
        test_prep_kernels_compile_once,
        test_decode_graph_replay,
        test_verify_graph_replay,
        test_topk_output_over_int32_elements,
        test_prefill_rejects_alias,
        test_decode_logits_padding_and_alias,
    ):
        test(device)
    for name in MALFORMED_DECODE:
        test_decode_rejects_malformed_inputs(name, device)
    for mode, axes in itertools.product(("prefill", "decode", "verify"), (1, 3)):
        test_yarn_prefill_decode_verify(mode, axes, device)
    test_prefill_capture_keeps_plan_after_eviction(device)
    for heads in (12, 6, 3):
        test_indexer_to_sparse_prefill(heads, device)
    for seq, ext, yarn in PREFILL_CAPTURE_CASES:
        test_prefill_graph_shapes(seq, ext, yarn, device)
    for lengths, window, context in DECODE_WIDE_GRAPH_CASES:
        test_decode_wide_graph(lengths, window, context, device)
    for window, yarn, graph in itertools.product(
        (1, 2, 3, 4), (False, True), (False, True)
    ):
        test_decode_state_trajectory(window, yarn, graph, device)
    for pattern, (counts, stride), split in itertools.product(
        ("random", "ties", "dense-threshold", "nan", "inf"),
        (((20000, 600, 65536, 513), 65536), ((70000, 600, 262144, 513, 65537), 262144)),
        (False, True),
    ):
        with pytest.MonkeyPatch.context() as patch:
            test_decode_topk_matches_row_kernel(
                pattern, counts, stride, split, device, patch
            )
    rows = []
    for mode, batch, context in itertools.product(
        args.modes, args.batch, args.contexts
    ):
        query_length = (
            min(context, 16384) if mode == "prefill" else (4 if mode == "verify" else 1)
        )
        rows.append(benchmark_indexer(mode, batch, context, query_length))
    aiter.logger.info(
        "QSA indexer summary (qk -> indices):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
