# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RDNA decode and prefill attention (aiter.ops.rdna_attn): every build of the
current GPU's tables against a float reference, and the calls the kernels
refuse. Runs on gfx1150..gfx1153 and RDNA4 (gfx1200, gfx1201); the table
checks run anywhere.

The first launch of a head dim and dtype builds its module, so a full run
builds one per (head dim, dtype) of the tables.

    pytest op_tests/test_rdna_attn.py
    pytest op_tests/test_rdna_attn.py -k "d64 and fp16"
"""

import functools
import itertools
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from aiter.ops import rdna_attn as hip

# Relative error bound per dtype, for decode.
TOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
# Absolute error bound per dtype, for prefill.
PREFILL_TOL = {torch.float16: 4e-3, torch.bfloat16: 2e-2}
DTYPES = list(TOL)


_CONFIG_ENTRY = re.compile(r"^\s*\{([\w:, ]+)\},$")


@pytest.mark.parametrize(
    "kind, shape_fields, causal_field",
    [
        ("decode", 7, None),
        ("decode_batch", 7, None),
        ("prefill", 6, 15),
        ("prefill_batch", 5, 9),
    ],
)
def test_gfx1150_configs_cover_the_same_shapes(kind, shape_fields, causal_field):
    """gfx1150's tables serve the shapes gfx1151's serve, but for the
    non-causal (mm-encoder) builds, gfx1151's only."""
    csrc = Path(__file__).resolve().parents[1] / "csrc" / "kernels" / "rdna_attn"
    text = (csrc / f"rdna_{kind}_configs.h").read_text()
    shapes = {}
    for table in text.split("inline constexpr ")[1:]:
        if name := re.match(r"\w+ k(Gfx\d+)\w*Configs\[\] = \{", table):
            entries = [
                m[1].split(", ")
                for m in map(_CONFIG_ENTRY.match, table.split("\n"))
                if m
            ]
            if causal_field is not None:
                entries = [
                    e
                    for e in entries
                    if len(e) <= causal_field or e[causal_field] != "0"
                ]
            shapes[name[1]] = {tuple(e[:shape_fields]) for e in entries}
    assert shapes["Gfx1151"] and shapes["Gfx1151"] == shapes["Gfx1150"]


def _on_rdna4() -> bool:
    return hip.device_family() == "GFX1201"


def _skip_unless_rdna35_or_rdna4():
    if hip.device_family() is None:
        pytest.skip("kernels are built for gfx1150..gfx1153 and RDNA4")


def _built(dtype=torch.float16, batch=0, **want):
    """The first built-in variant with these properties, loaded.

    Args:
        dtype: Element type.
        batch: 1 for a batched build.
        **want: Predicates on KernelVariant fields, by name.

    Returns:
        (variant, module).

    """
    _skip_unless_rdna35_or_rdna4()
    for v in hip.variants():
        if (
            v.dtype == dtype
            and v.batched == batch
            and all(pred(getattr(v, k)) for k, pred in want.items())
        ):
            module = hip.load(v)
            assert module is not None, f"{v.name} is not in the tables"
            return v, module
    raise AssertionError(f"no built-in variant with {sorted(want)}")


def _paged_inputs(seq_len, v, seed=0):
    """An HND KV cache whose last page is partly used."""
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
    kv = kv * 0.5
    q = torch.randn(
        v.max_query_len, v.num_q_heads, v.head_size, device=dev, dtype=v.dtype
    )
    return q * 0.5, kv, torch.arange(num_blocks, device=dev, dtype=torch.int32)


def _reference(q, kv, seq_len, window=0, mm_ranges=()):
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    flat = kv.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len]
    k, v = flat[..., :hd], flat[..., hd:]
    gqa = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (hd**-0.5)
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
    for start, end in mm_ranges:
        masked &= ~((lim >= start) & (lim <= end) & (pos >= start) & (pos <= end))
    if window:
        masked |= pos < lim - (window - 1)
    scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
    return torch.bmm(torch.softmax(scores, -1), vf).permute(1, 0, 2)


def _rounds_p(v) -> bool:
    """On RDNA4, fp16 WMMA decode builds with more than 8 rows per workgroup
    take P rounded to fp16 into P@V."""
    if not isinstance(v, hip.KernelVariant):
        return False
    gqa = v.num_q_heads // v.num_kv_heads
    rows = gqa * v.max_query_len // v.row_groups
    return _on_rdna4() and v.dtype == torch.float16 and not v.dot_product and rows > 8


def _max_rel(got, ref, v=None) -> float:
    """Relative error with a floor. With P rounded to fp16 every term errs by
    up to 2^-11, so outputs near zero are judged against the largest one."""
    floor = 1e-3
    if v is not None and _rounds_p(v):
        floor = max(floor, ref.abs().max().item())
    return ((got - ref).abs() / ref.abs().clamp_min(floor)).max().item()


def _launch(module, q, kv, block_table, out, scratch, seq_len):
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5)


def _check_one(v, module, lens):
    """Launches sharing one scratch, each on new inputs."""
    scratch = hip.make_scratch(v, torch.device("cuda"))
    for seed, seq_len in enumerate(lens):
        q, kv, block_table = _paged_inputs(seq_len, v, seed)
        out = torch.full_like(q, float("nan"))
        _launch(module, q, kv, block_table, out, scratch, seq_len)
        torch.cuda.synchronize()
        ref = _reference(q, kv, seq_len, v.window)
        got = out.float().nan_to_num(1e9)
        assert _max_rel(got, ref, v) <= TOL[v.dtype], f"{v.name}, S={seq_len}"


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


def _check_batch(v, module, lens):
    """Check a batched build over these lengths, twice on one scratch; a
    padded sequence (S=0) stays unwritten."""
    parts, q, kv, block_table = _batch_inputs(v, lens)
    scratch = hip.make_scratch(v, q.device, max_seqs=len(lens))
    seq_lens = torch.tensor(lens, device=q.device, dtype=torch.int32)
    out = torch.full_like(q, 7.0)
    for _ in range(2):
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5
        )
    torch.cuda.synchronize()
    for i, s in enumerate(lens):
        rows = out[i * v.max_query_len : (i + 1) * v.max_query_len].float()
        if s == 0:
            assert (rows == 7.0).all(), f"{v.name}: a padded sequence was written"
            continue
        ref = _reference(parts[i][0], parts[i][1], s, v.window)
        assert _max_rel(rows, ref, v) <= TOL[v.dtype], f"{v.name}: sequence {i}, S={s}"


def _unit(v):
    """A test case: the variants of one shape and dtype."""
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return f"q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"


def test_decode_configs_are_well_formed():
    """No two decode builds share a shape, and every built shape covers every
    M in both dtypes, for one sequence and a batch."""
    variants = hip.variants()
    if not variants:
        pytest.skip("no RDNA tables for this GPU")
    assert len({hip._lookup_key(v) for v in variants}) == len(
        variants
    ), "two builds share a shape"
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window, v.page_size)
        for v in variants
    }
    for hq, hkv, d, window, bs in configs:
        assert d in hip.SUPPORTED_HEAD_SIZES and hq % hkv == 0
        for m, dtype, batch in itertools.product(
            range(1, hip.MAX_M + 1), DTYPES, (False, True)
        ):
            v = hip.variant_for(hq, hkv, d, m, bs, 1, window, dtype, batch)
            assert v is not None, (hq, hkv, d, window, bs, m, dtype, batch)
            assert not (batch and v.dot_product), v.name


def test_every_listed_variant_loads():
    _skip_unless_rdna35_or_rdna4()
    assert hip.rdna_decode_configs(torch.cuda.current_device())
    assert hip.rdna_decode_batch_configs(torch.cuda.current_device())
    missing = [v.name for v in hip.variants() if hip.load(v) is None]
    assert not missing, f"{len(missing)} variants not loadable, e.g. {missing[0]}"


@pytest.mark.parametrize("unit", sorted({_unit(v) for v in hip.variants()}))
def test_built_in_variants_match_reference(unit):
    """Every built decode variant matches the reference."""
    _skip_unless_rdna35_or_rdna4()
    for v in hip.variants():
        if _unit(v) != unit:
            continue
        module = hip.load(v)
        assert module is not None, v.name
        s = max(1024, v.window + 64, v.page_size + 1000)
        if v.batched:
            _check_batch(v, module, [s, 48, 0])
            continue
        _check_one(v, module, [48, s, s + 1000])
        # An image ending among the query tokens: rows on each side differ.
        m = v.max_query_len
        ranges = [(s - m - 40, s - m + m // 2)]
        q, kv, block_table = _paged_inputs(s, v)
        buf = _decode_mm_buf(ranges, q.device)
        got = _decode_mm(module, v, q, kv, block_table, s, buf)
        ref = _reference(q, kv, s, v.window, ranges)
        assert _max_rel(got, ref, v) <= TOL[v.dtype], f"{v.name}, image"
        # At M=1 the image ends at the only query token, so nothing changes.
        if m > 1:
            causal = _reference(q, kv, s, v.window)
            assert _max_rel(causal, ref) > 10 * TOL[v.dtype], f"{v.name}, image"


@pytest.mark.parametrize(
    "want",
    [
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
    ],
    ids=["wmma segments", "wmma", "dot_product"],
)
def test_sliding_window_never_reads_freed_pages(want):
    """Freed pages before the window do not reach the output."""
    v, module = _built(**want)
    seq_len = 2016
    q, kv, block_table = _paged_inputs(seq_len, v)
    ref = _reference(q, kv, seq_len, v.window)
    first = max(0, seq_len - v.max_query_len - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    _launch(module, q, kv, block_table, out, hip.make_scratch(v, q.device), seq_len)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert _max_rel(out.float(), ref, v) <= TOL[v.dtype]


def test_graph_replay_follows_seq_lens():
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
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(q, kv, s)
        assert _max_rel(out.float(), ref, v) <= TOL[v.dtype], f"S={s}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_other_dtype_is_refused_not_miscomputed(dtype):
    """A build refuses the other 16-bit dtype: its module holds no kernel of
    that dtype."""
    other = next(d for d in DTYPES if d != dtype)
    v, module = _built(dtype=other)
    q, kv, block_table = _paged_inputs(1024, v)
    q, kv = q.to(dtype), kv.to(dtype)
    with pytest.raises(RuntimeError, match="not built for its head dim, dtype"):
        _launch(
            module,
            q,
            kv,
            block_table,
            torch.empty_like(q),
            hip.make_scratch(v, q.device),
            1024,
        )


def test_build_of_another_head_dim_is_refused():
    """A module launches only the builds of its head dim and dtype."""
    v, _ = _built()
    other = next(
        o for o in hip.variants() if o.head_size != v.head_size and not o.batched
    )
    q, kv, block_table = _paged_inputs(64, v)
    seq_lens = torch.tensor([64], device=q.device, dtype=torch.int32)
    index = hip._builds(0).index(other)
    with pytest.raises(RuntimeError, match="not built for its head dim"):
        hip.rdna_decode_attn(
            q, kv, block_table, torch.empty_like(q), *hip.make_scratch(v, q.device),
            seq_lens, index, 0.1,
        )  # fmt: skip


# ---------------------------------------------------------------- prefill


def _prefill_reference(
    q, kv, block_table, seq_len, window=0, mm_ranges=(), causal=True
):
    """Float reference of a prefill; a token in a multimodal range [start, end]
    also sees the range's later keys. Non-causal: every key."""
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    pages = kv[block_table.long()]
    flat = pages.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len].float()
    gqa = hq // hkv
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim if causal else torch.zeros_like(pos > lim)
    for start, end in mm_ranges:
        q_in = (lim >= start) & (lim <= end)
        k_in = (pos >= start) & (pos <= end)
        masked &= ~(q_in & k_in)
    if window:
        masked |= pos < lim - (window - 1)
    out = torch.empty(m, hq, hd, device=q.device)
    for h0 in range(0, hq, 4):
        heads = range(h0, min(hq, h0 + 4))
        kvh = [h // gqa for h in heads]
        qf = q[:, h0 : h0 + len(heads)].float().permute(1, 0, 2)
        k = flat[:, kvh, :hd].permute(1, 0, 2)
        v = flat[:, kvh, hd:].permute(1, 0, 2)
        scores = torch.bmm(qf, k.transpose(1, 2)) * (hd**-0.5)
        scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
        out[:, h0 : h0 + len(heads)] = torch.bmm(torch.softmax(scores, -1), v).permute(
            1, 0, 2
        )
    return out


def _mm_ranges_tensor(mm_ranges, device):
    """One sequence's ranges as (1, ranges, 2), padded with an
    empty range."""
    pairs = [*mm_ranges, (0, 0)]
    return torch.tensor([pairs], device=device, dtype=torch.int32)


def _assert_prefill_close(got, q, kv, block_table, seq_len, window, what, mm_ranges=()):
    """Against the float reference."""
    ref = _prefill_reference(q, kv, block_table, seq_len, window, mm_ranges)
    err = (got.float().nan_to_num(1e9) - ref).abs().max().item()
    bound = PREFILL_TOL[q.dtype]
    assert err <= bound, f"{what}: max error {err:.2e}, bound {bound:.2e}"


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


def _prefill_unit(v):
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    mode = "" if v.causal else "_noncausal"
    return (
        f"prefill_q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"
        f"{mode}"
    )


def _prefill_rows():
    return [v for rows in hip._prefill_table().values() for v in rows]


def _causal_prefill_rows():
    return [v for v in _prefill_rows() if v.causal]


def test_prefill_configs_are_well_formed():
    """No two prefill builds share a shape and step, every decode configuration
    has prefill builds, and the build of the largest step reached runs."""
    if not _prefill_rows():
        pytest.skip("no RDNA tables for this GPU")
    keys = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        + (v.dtype, v.min_query_len, v.causal)
        for v in _prefill_rows()
    }
    assert len(keys) == len(_prefill_rows()), "two builds share a step"
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window)
        for v in hip.variants()
        if v.page_size == 16
    }
    for (hq, hkv, d, window), dtype in itertools.product(configs, DTYPES):
        steps = hip.prefill_variants(hq, hkv, d, 16, window, dtype)
        assert steps, (hq, hkv, d, window, dtype)
        starts = [v.min_query_len for v in steps]
        assert starts == sorted(set(starts))
        for v in steps:
            for m in (v.min_query_len, v.min_query_len + 1):
                assert hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m) == v
    v = steps[0]
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, 16, 1, v.window, v.dtype)
    assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M - 1) is None
    assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M) == steps[0]
    assert hip.prefill_variant_for(*args, 1 << 20) == steps[-1]


def test_every_listed_prefill_variant_loads():
    _skip_unless_rdna35_or_rdna4()
    assert hip.rdna_prefill_configs(torch.cuda.current_device())
    missing = [v for v in _prefill_rows() if hip.load_prefill(v) is None]
    assert not missing, f"{len(missing)} prefill variants not loadable"


@pytest.mark.parametrize(
    "unit", sorted({_prefill_unit(v) for v in _causal_prefill_rows()})
)
def test_built_in_prefill_variants_match_reference(unit):
    """Every built prefill variant matches the reference, with and without a
    prefix."""
    _skip_unless_rdna35_or_rdna4()
    for v in _prefill_rows():
        if _prefill_unit(v) != unit:
            continue
        module = hip.load_prefill(v)
        assert module is not None, v
        args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        scratch = hip.make_prefill_scratch(
            hip.prefill_variants(*args, v.dtype), torch.device("cuda")
        )
        prefix = max(4096, 2 * v.window)
        cases = [(v.min_query_len, 0), (v.min_query_len + 37, prefix)]
        if v == hip.prefill_variants(*args, v.dtype)[0]:
            cases += [(hip.PREFILL_MIN_M, prefix), (hip.PREFILL_MIN_M + 50, 0)]
        for seed, (m, p) in enumerate(cases):
            seq_len = m + p
            q, kv, block_table = _prefill_inputs(v, m, seq_len, seed)
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
                v.head_size**-0.5,
            )
            torch.cuda.synchronize()
            _assert_prefill_close(
                out, q, kv, block_table, seq_len, v.window, f"{v}, M={m}, prefix={p}"
            )


def test_prefill_past_the_last_step():
    """M past the last step runs the last row."""
    _skip_unless_rdna35_or_rdna4()
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
        _assert_prefill_close(
            out, q, kv, block_table, seq_len, v.window, f"{v}, M={m}, prefix={p}"
        )


def test_prefill_window_never_reads_freed_pages():
    """Freed pages before the window do not reach the output."""
    _skip_unless_rdna35_or_rdna4()
    v = next(v for v in _prefill_rows() if v.window and v.dtype == torch.float16)
    module = hip.load_prefill(v)
    m, seq_len = v.min_query_len, v.min_query_len + 3 * v.window
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    clean = (kv.clone(), block_table.clone())
    first = (seq_len - m - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(hip.prefill_variants(*args, v.dtype), q.device)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    _assert_prefill_close(out, q, *clean, seq_len, v.window, "freed pages")


# ---------------------------------------------------------------- non-causal


def _non_causal_rows():
    return [v for v in _prefill_rows() if not v.causal]


def _non_causal_inputs(v, lens, seed=0):
    """Q, K and V of sequences of `lens` tokens, one after the other, as views
    of one (tokens, 3, H, D) projection (a token stride of 3 H D)."""
    torch.manual_seed(seed)
    n = sum(lens)
    qkv = torch.randn(n, 3, v.num_q_heads, v.head_size, device="cuda") * 0.5
    qkv = qkv.to(v.dtype)
    q, k, val = qkv[:, 0], qkv[:, 1, : v.num_kv_heads], qkv[:, 2, : v.num_kv_heads]
    return q, k, val, np.cumsum([0] + list(lens))


def _assert_non_causal_close(got, q, k, v, what):
    """Against the float reference, for one sequence."""
    m, hq, hd = q.shape
    g = hq // k.shape[1]
    # A few heads at a time: all 64 heads of a long sequence take 16 GiB.
    step = max(1, (1 << 28) // (m * m))
    ref = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                *(
                    x[:, h : h + step].float().transpose(0, 1)
                    for x in (q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1))
                ),
                scale=hd**-0.5,
            ).transpose(0, 1)
            for h in range(0, hq, step)
        ],
        dim=1,
    )
    err = (got.float().nan_to_num(1e9) - ref).abs().max().item()
    bound = PREFILL_TOL[q.dtype]
    assert err <= bound, f"{what}: max error {err:.2e}, bound {bound:.2e}"


@pytest.mark.parametrize("unit", sorted({_prefill_unit(v) for v in _non_causal_rows()}))
def test_built_in_non_causal_prefill_variants_match_reference(unit):
    """Every non-causal build: each token sees every key of its sequence,
    read in place, also past the last whole step, including a last step whose
    later 16-key sub-tiles hold no key."""
    _skip_unless_rdna35_or_rdna4()
    for v in _non_causal_rows():
        if _prefill_unit(v) != unit:
            continue
        module = hip.load_prefill(v)
        assert module is not None, v
        args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        rows = hip.prefill_variants(*args, v.dtype, causal=False)
        scratch = hip.make_prefill_scratch(rows, torch.device("cuda"))
        whole = -(-v.min_query_len // v.key_tile) * v.key_tile
        lens = [v.min_query_len, v.min_query_len + 37, whole + 8, whole + 16]
        if v == rows[0]:
            lens += [hip.PREFILL_MIN_M, 100]
        for seed, m in enumerate(lens):
            q, k, val, _ = _non_causal_inputs(v, [m], seed)
            out = torch.full(q.shape, float("nan"), dtype=q.dtype, device=q.device)
            seq_lens = torch.tensor([m], device=q.device, dtype=torch.int32)
            module.encoder_attn(
                q, k, val, out, *scratch, seq_lens, m, v.head_size**-0.5
            )
            torch.cuda.synchronize()
            _assert_non_causal_close(out, q, k, val, f"{v}, L={m}")


_NON_CAUSAL_BATCHES = {
    "5 x 577": [577] * 5,
    "mixed": [1024, 257, 9, 100, 64],
    "windows": [64] * 20 + [16, 36, 48],
}


@pytest.mark.parametrize("batch", _NON_CAUSAL_BATCHES)
@pytest.mark.parametrize(
    "shape",
    sorted(
        {
            (v.num_q_heads, v.num_kv_heads, v.head_size, v.dtype)
            for v in _non_causal_rows()
        },
        key=str,
    ),
    ids=str,
)
def test_non_causal_prefill_batch_matches_reference(shape, batch):
    """An encoder's segments in one planned launch, each against its
    reference."""
    _skip_unless_rdna35_or_rdna4()
    hq, hkv, d, dtype = shape
    lens = _NON_CAUSAL_BATCHES[batch]
    dev = torch.device("cuda")
    seq_lens = np.array(lens)
    v = hip.prefill_variants(hq, hkv, d, 16, 0, dtype, causal=False)[0]
    q, k, val, starts = _non_causal_inputs(v, lens)
    v, head_group = hip.prefill_batch_variant_for(
        hq, hkv, d, 16, 1, 0, dtype, starts, seq_lens, causal=False
    )
    plan, head_group = hip.plan_prefill(
        starts, seq_lens, v, 0, hip.prefill_target_workgroups(dev), head_group
    )
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(hq, hkv, d, 16, 0, dtype, causal=False), dev
    )
    out = torch.full(q.shape, float("nan"), dtype=dtype, device=dev)
    hip.load_prefill(v).encoder_attn(
        q, k, val, out, *scratch,
        torch.tensor(lens, device=dev, dtype=torch.int32), max(lens), d**-0.5,
        plan=torch.from_numpy(plan).to(dev),
        query_start_loc=torch.from_numpy(starts.astype(np.int32)).to(dev),
        head_group=head_group,
    )  # fmt: skip
    torch.cuda.synchronize()
    for i, n in enumerate(lens):
        rows = slice(int(starts[i]), int(starts[i + 1]))
        _assert_non_causal_close(
            out[rows], q[rows], k[rows], val[rows], f"{batch}: sequence {i} (L={n})"
        )


def test_non_causal_rows_are_full_attention():
    """Non-causal builds have no window and run only through encoder_attn;
    causal builds only through prefill_attn."""
    rows = _non_causal_rows()
    if not rows:
        pytest.skip("no non-causal RDNA builds for this GPU")
    assert all(v.window == 0 for v in rows)
    _skip_unless_rdna35_or_rdna4()
    dev = torch.device("cuda")
    causal_sizes = {c.head_size for c in _causal_prefill_rows()}
    v = next(v for v in rows if v.head_size in causal_sizes)
    scratch = hip.make_prefill_scratch([v], dev)
    q, k, val, _ = _non_causal_inputs(v, [64])
    seq_lens = torch.tensor([64], device=dev, dtype=torch.int32)
    kv = torch.zeros(4, v.num_kv_heads, 16, 2 * v.head_size, dtype=v.dtype, device=dev)
    table = torch.arange(4, dtype=torch.int32, device=dev)
    out = torch.empty(q.shape, dtype=q.dtype, device=dev)
    with pytest.raises(RuntimeError, match="rdna_encoder_attn"):
        hip.load_prefill(v).prefill_attn(
            q.contiguous(), kv, table, out, *scratch, seq_lens, 64, 0.1
        )
    causal = next(c for c in _causal_prefill_rows() if c.head_size == v.head_size)
    with pytest.raises(RuntimeError, match="rdna_prefill_attn"):
        hip.load_prefill(causal).encoder_attn(
            q, k, val, out, *scratch, seq_lens, 64, 0.1
        )


@pytest.mark.parametrize(
    "segments",
    [[4, 8, 1, 3], [577, 5, 1024, 8, 9, 64, 8], [9] * 7, [4096], [5], []],
    ids=["short", "mixed", "first step", "one", "one short", "none"],
)
def test_encoder_forward_matches_reference(segments):
    """encoder_forward: every segment, however short, in one non-causal
    launch."""
    _skip_unless_rdna35_or_rdna4()

    shapes = sorted(
        {
            (v.num_q_heads, v.num_kv_heads, v.head_size, v.dtype)
            for v in _non_causal_rows()
        },
        key=str,
    )
    if not shapes:
        pytest.skip("no non-causal RDNA builds for this GPU")
    torch.manual_seed(0)
    for hq, hkv, d, dtype in shapes:
        n = sum(segments)
        q, k, v = (
            (torch.randn(n, h, d, device="cuda") * 0.5).to(dtype)
            for h in (hq, hkv, hkv)
        )
        assert hip.encoder_unsupported(q, hkv) is None
        out = hip.encoder_forward(q, k, v, segments, d**-0.5)
        start = 0
        for i, s in enumerate(segments):
            rows = slice(start, start + s)
            ref = torch.nn.functional.scaled_dot_product_attention(
                *(x[rows].float().transpose(0, 1) for x in (q, k, v)),
                scale=d**-0.5,
                enable_gqa=hq != hkv,
            ).transpose(0, 1)
            err = (out[rows].float() - ref).abs().max().item()
            tol = PREFILL_TOL[dtype]
            assert err <= tol, f"{hq}/{hkv}/{d} {dtype}: segment {i} (L={s}) {err:.2e}"
            start += s


# ---------------------------------------------------------------- prefill batches

# (M, S) per sequence.
_BATCHES = {
    "decode+chunk+prompt": [(1, 6000), (300, 2348), (40, 40)],
    "8 prompts": [(256, 256)] * 8,
    "unequal decodes": [(1, 37), (3, 1000)],
    "padded": [(64, 1064), (0, 0), (17, 17)],
}


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


@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("batch", [*_BATCHES, "at the bound"])
def test_prefill_plan_covers_every_tile_once(batch, window):
    """The work list covers every tile once, heaviest first, within the
    scratch's bounds."""
    target = 160
    v = hip.prefill_variant_for(32, 8, 128, 16, 1, 0, torch.float16, 128)
    if v is None:
        pytest.skip("no RDNA tables for this GPU")
    # The most counters and partials the bounds allow.
    lens = (
        [(1, 32768)] * 19 + [(16, 16)] if batch == "at the bound" else _BATCHES[batch]
    )
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
    if batch == "at the bound" and not window:
        assert len(split_tiles) == 19


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


def _serve_batch(v, q, kv, block_table, starts, seq_lens, out, mm_ranges=None):
    """A batch in one planned launch, on the build and head group of its
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
    hip.load_prefill(build).prefill_attn(
        q,
        kv,
        block_table,
        out,
        *scratch,
        seq_lens.to(dev),
        int(seq_lens.max()),
        v.head_size**-0.5,
        ranges,
        plan=torch.from_numpy(plan).to(dev),
        query_start_loc=starts.to(dev),
        head_group=head_group,
    )


@pytest.mark.parametrize(
    "shape",
    [(32, 16, 256, 1024, torch.float16), (32, 4, 512, 0, torch.bfloat16)],
    ids=["swa_fp16", "global_bf16"],
)
def test_prefill_batch_mm_prefix_matches_reference(shape):
    """A batch with each sequence's own multimodal ranges."""
    _skip_unless_rdna35_or_rdna4()
    hq, hkv, d, window, dtype = shape
    v = hip.prefill_variants(hq, hkv, d, 16, window, dtype)[0]
    lens = [(600, 600), (1, 2000), (700, 3700), (300, 800)]
    ranges = [[(40, 295), (400, 469)], [], [(3100, 3355)], [(450, 560)]]
    q, kv, block_table, starts, seq_lens = _prefill_batch(v, lens)
    out = torch.full_like(q, float("nan"))
    _serve_batch(v, q, kv, block_table, starts, seq_lens, out, mm_ranges=ranges)
    torch.cuda.synchronize()
    for i, (m, s) in enumerate(lens):
        rows = slice(int(starts[i]), int(starts[i + 1]))
        _assert_prefill_close(
            out[rows],
            q[rows],
            kv,
            block_table[i, : -(-s // v.page_size)],
            s,
            window,
            f"sequence {i} (M={m}, S={s})",
            ranges[i],
        )


@pytest.mark.parametrize(
    "unit",
    sorted({_prefill_unit(v) for v in _causal_prefill_rows() if v.page_size == 16}),
)
def test_prefill_batch_matches_reference(unit):
    """A batch in one launch, each sequence against its reference."""
    _skip_unless_rdna35_or_rdna4()
    v = next(v for v in _prefill_rows() if _prefill_unit(v) == unit)
    for seed, (name, lens) in enumerate(_BATCHES.items()):
        lens = [(m, s + v.window if m != s else s) for m, s in lens]
        q, kv, block_table, starts, seq_lens = _prefill_batch(v, lens, seed)
        out = torch.full_like(q, float("nan"))
        _serve_batch(v, q, kv, block_table, starts, seq_lens, out)
        torch.cuda.synchronize()
        for i, (m, s) in enumerate(lens):
            if not m:
                continue
            rows = slice(int(starts[i]), int(starts[i + 1]))
            _assert_prefill_close(
                out[rows],
                q[rows],
                kv,
                block_table[i, : -(-s // v.page_size)],
                s,
                v.window,
                f"{unit} {name}: sequence {i} (M={m}, S={s})",
            )


def test_prefill_batch_configs_are_well_formed():
    """Every cell names a prefill build of its shape and a head group that
    divides its kv heads; each shape has the first cell."""
    table = hip._prefill_batch_table()
    if not table:
        pytest.skip("no RDNA tables for this GPU")
    seen = set()
    for (hq, hkv, d, page, window, causal), cells in table.items():
        for mean_m, keys, min_query_len, head_group in cells:
            key = (hq, hkv, d, page, window, causal, mean_m, keys)
            assert key not in seen, key
            seen.add(key)
            assert mean_m in hip.PREFILL_BATCH_MEAN_M
            assert keys in hip.PREFILL_BATCH_KEYS
            assert head_group == 0 or hkv % head_group == 0
            for dtype in DTYPES:
                steps = hip.prefill_variants(hq, hkv, d, page, window, dtype, causal)
                if steps or causal:
                    assert min_query_len in {v.min_query_len for v in steps}, key
    for config in {k[:6] for k in seen}:
        assert (*config, hip.PREFILL_BATCH_MEAN_M[0], 0) in seen, config


def test_prefill_batch_cell_lookup(monkeypatch):
    """A batch takes the cell of the largest steps at or below its features."""
    table = {
        (32, 8, 128, 16, 0, 1): [(1, 0, 16, 8), (1, 2048, 128, 1), (256, 512, 512, 4)]
    }
    monkeypatch.setattr(hip, "_prefill_batch_table", lambda: table)
    cell = functools.partial(hip.prefill_batch_cell, 32, 8, 128, 16, 0)
    assert cell(4, 100) == (16, 8)
    assert cell(4, 5000) == (128, 1)
    assert cell(300, 5000) == (512, 4)
    assert cell(300, 100) == (512, 4)  # no keys step at or below: the smallest
    assert hip.prefill_batch_cell(16, 2, 64, 16, 0, 4, 100) is None


def test_prefill_batch_features():
    """Padding left out; keys capped at the window."""
    starts = np.array([0, 4, 4, 12])
    seq_lens = np.array([100, 0, 8])
    mean_m, keys = hip.prefill_batch_features(starts, seq_lens, 0)
    assert mean_m == 6
    # Sequence 0: context 96, keys 96 + 2.5; sequence 1: context 0, keys 4.5.
    assert keys == pytest.approx((4 * 98.5 + 8 * 4.5) / 12)
    assert hip.prefill_batch_features(starts, seq_lens, 10)[1] == pytest.approx(
        (4 * 10 + 8 * 4.5) / 12
    )


@pytest.mark.parametrize(
    "shape",
    [(32, 16, 256, 1024, torch.float16), (32, 4, 512, 0, torch.float16)]
    + [(32, 16, 256, 1024, torch.bfloat16)],
    ids=["swa_fp16", "global_fp16", "swa_bf16"],
)
@pytest.mark.parametrize(
    "m, prefix, mm_ranges",
    [
        (600, 0, [(40, 295), (400, 469)]),
        (700, 3000, [(3100, 3355)]),
        (300, 500, [(450, 560)]),
        # Longer than the 1024-key window: the window bounds earlier keys.
        (2048, 512, [(600, 2099)]),
    ],
    ids=["two_images", "split_kv", "from_prefix", "longer_than_window"],
)
def test_prefill_mm_prefix_matches_reference(shape, m, prefix, mm_ranges):
    """Tokens of a multimodal range see the whole range (Gemma4's shapes)."""
    _skip_unless_rdna35_or_rdna4()
    hq, hkv, d, window, dtype = shape
    v = hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m)
    assert v is not None, shape
    module = hip.load_prefill(v)
    assert module is not None, v
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
    _assert_prefill_close(
        out, q, kv, block_table, seq_len, window, f"{v}, M={m}", mm_ranges
    )


@pytest.mark.parametrize("bad", ["batch", "triples", "cpu"])
def test_prefill_refuses_malformed_mm_ranges(bad):
    """Only one sequence's (ranges, 2) on the device reaches the kernel: the
    batch's (sequences, ranges, 2) would unmask other sequences' keys."""
    _skip_unless_rdna35_or_rdna4()
    v = hip.prefill_variant_for(32, 16, 256, 16, 1, 1024, torch.float16, 64)
    module = hip.load_prefill(v)
    scratch = hip.make_prefill_scratch([v], torch.device("cuda"))
    q, kv, block_table = _prefill_inputs(v, 64, 64)
    good = _mm_ranges_tensor([(10, 40)], q.device)
    ranges = {
        "batch": good,
        "triples": torch.zeros(2, 3, dtype=torch.int32, device=q.device),
        "cpu": good[0].cpu(),
    }[bad]
    seq_lens = torch.tensor([64], device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mm_ranges must be"):
        module.prefill_attn(
            q, kv, block_table, torch.empty_like(q), *scratch, seq_lens, 64,
            256**-0.5, ranges,
        )  # fmt: skip


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
        q, kv, block_table, out, *hip.make_scratch(v, q.device), seq_lens,
        q.shape[2] ** -0.5, buf,
    )  # fmt: skip
    torch.cuda.synchronize()
    return out.float().nan_to_num(1e9)


@pytest.mark.parametrize(
    "want, min_m",
    [
        (
            {
                "window": bool,
                "dot_product": lambda x: not x,
                "max_segments": lambda x: x > 1,
            },
            4,
        ),
        (
            {
                "window": lambda x: not x,
                "dot_product": lambda x: not x,
                "max_segments": lambda x: x > 1,
            },
            4,
        ),
        (
            {
                "window": lambda x: not x,
                "dot_product": lambda x: not x,
                "max_segments": lambda x: x == 1,
            },
            4,
        ),
        # Dot-product builds go up to M=3; an image starting among M=2's query
        # tokens would only hold the last one.
        ({"dot_product": bool}, 3),
    ],
    ids=["wmma_window segments", "wmma segments", "wmma", "dot_product"],
)
@pytest.mark.parametrize(
    "image",
    [
        # Over every query token, beside a past image.
        lambda s, m, w: [(100, 355), (s - m - 40, s - 1)],
        # Ending, or starting, among the query tokens: the rows after it, or
        # before it, stay causal.
        lambda s, m, w: [(s - m - 40, s - m + m // 2)],
        lambda s, m, w: [(s - m + m // 2, s - 1)],
        # Past the sequence, as in a chunk ending inside the image.
        lambda s, m, w: [(s - m - 40, s + 200)],
        # Longer than the window, which still bounds earlier keys.
        lambda s, m, w: [(s - (w or 1000) - 500, s - 1)],
        # Two images over the query tokens: rows end at either one.
        lambda s, m, w: [(s - m - 40, s - m + 1), (s - m + 2, s - 1)],
    ],
    ids=[
        "covers_all",
        "ends_mid",
        "starts_mid",
        "past_seq_len",
        "longer_than_window",
        "two_in_span",
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_decode_mm_prefix_matches_reference(want, min_m, image, dtype):
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
    ref = _reference(q, kv, seq_len, v.window, ranges)
    assert _max_rel(got, ref, v) <= TOL[v.dtype], v.name
    causal = _reference(q, kv, seq_len, v.window)
    assert _max_rel(causal, ref) > 10 * TOL[v.dtype], "the image changed nothing"
    # With several images over the query tokens, each one shows.
    cut = [r for r in ranges if r[0] < r[1] and r[1] >= seq_len - m]
    for r in cut if len(cut) > 1 else []:
        rest = [x for x in ranges if x != r]
        without = _reference(q, kv, seq_len, v.window, rest)
        assert _max_rel(without, ref) > 10 * TOL[v.dtype], f"{r} changed nothing"


@pytest.mark.parametrize("case", ["zero", "over_capacity"])
def test_decode_mm_ranges_stop_at_count_and_capacity(case):
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
    ref = _reference(q, kv, seq_len, 0, ranges)
    assert _max_rel(got, ref, v) <= TOL[v.dtype], v.name
    # Reading the trap, or ignoring the ranges that fit, must show.
    for wrong in (ranges + [trap], []):
        if wrong != ranges:
            wrong_ref = _reference(q, kv, seq_len, 0, wrong)
            assert _max_rel(wrong_ref, ref) > 10 * TOL[v.dtype], wrong


def test_graph_replay_follows_mm_ranges():
    """A captured graph reads the ranges the builder rewrites in place each
    step, and none are left over when they go back to empty."""
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
        # As the builder: a smaller count leaves earlier ranges past it.
        flat = _decode_mm_buf(ranges, q.device)[: 1 + 2 * len(ranges)]
        buf[: flat.numel()].copy_(flat)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(q, kv, seq_len, 0, ranges)
        assert _max_rel(out.float(), ref, v) <= TOL[v.dtype], f"ranges={ranges}"


def test_batched_decode_follows_mm_ranges_per_sequence():
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
    # A spare row, as the builder's buffer has one per max_num_seqs.
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
    steps: list[tuple[list[int], list[list[tuple[int, int]]]]] = [
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
                ref = _reference(parts[i][0], parts[i][1], s, v.window, ranges)
                assert _max_rel(got, ref, v) <= TOL[v.dtype], f"{step}, sequence {i}"
                if ranges:
                    causal = _reference(parts[i][0], parts[i][1], s, v.window)
                    assert _max_rel(causal, ref) > 10 * TOL[v.dtype], "no change"


@pytest.mark.parametrize("bad", ["even", "int64", "cpu", "no_rows"])
def test_decode_refuses_malformed_mm_ranges(bad):
    """Only an odd-length int32 [count, start0, end0, ...], or a row of it per
    sequence, on the device reaches the kernel, which sizes its capacity from
    it."""
    v, module = _built()
    q, kv, block_table = _paged_inputs(64, v)
    good = _decode_mm_buf([(10, 40)], q.device)
    ranges = {
        "even": good[:4],
        "int64": good.long(),
        "cpu": good.cpu(),
        "no_rows": good.view(1, -1)[:0],
    }[bad]
    seq_lens = torch.tensor([64], device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mm_ranges must be"):
        module.decode_attn(
            q, kv, block_table, torch.empty_like(q), *hip.make_scratch(v, q.device),
            seq_lens, q.shape[2] ** -0.5, ranges,
        )  # fmt: skip


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
