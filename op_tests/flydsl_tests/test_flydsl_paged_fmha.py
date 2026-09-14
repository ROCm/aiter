# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Native paged FP8 prefill contracts and independent numerical references."""

from types import SimpleNamespace

import pytest
import torch

from aiter.ops.flydsl import flydsl_flash_attn_paged_fp8_func
from aiter.ops.flydsl.kernels import flash_attn_paged_fp8_func_gfx950 as paged

LAYOUTS = [
    (1, "linear3d"),
    (1, "linear"),
    (16, "vectorized"),
    (64, "vectorized"),
    (1024, "vectorized"),
]
DIMS = [(128, 128), (192, 128), (192, 192)]
_arch = (
    torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if torch.cuda.is_available()
    else ""
)
gfx950 = pytest.mark.skipif(
    _arch != "gfx950", reason="native paged FP8 requires gfx950"
)


def _quantize(x):
    scale = (x.abs().max() / 448).clamp_min(1e-8).reshape(1)
    return (x / scale).to(torch.float8_e4m3fn), scale


def make_case(
    page,
    layout,
    d,
    dv,
    *,
    qlens=(300, 65, 17),
    klens=(513, 193, 81),
    heads=(6, 2),
    seed=17,
):
    """Build independent physical caches and a packed query batch."""
    torch.manual_seed(seed)
    hq, hkv = heads
    counts = [(n + page - 1) // page for n in klens]
    maxkv = ((max(klens, default=0) + 127) // 128) * 128
    capacity = (maxkv + page - 1) // page
    npages = max(1, sum(counts))
    ids = torch.randperm(npages, device="cuda", dtype=torch.int32)
    table = torch.full((len(qlens), capacity), -1, device="cuda", dtype=torch.int32)
    offset = 0
    for b, count in enumerate(counts):
        table[b, :count] = ids[offset : offset + count]
        offset += count
    q, qs = _quantize(torch.randn(max(sum(qlens), 1), hq, d, device="cuda") * 0.2)
    q = q[: sum(qlens)]
    klin, ks = _quantize(torch.randn(npages, page, hkv, d, device="cuda") * 0.2)
    vlin, vs = _quantize(torch.randn(npages, page, hkv, dv, device="cuda") * 0.2 + 0.25)
    if layout == "vectorized":
        k = (
            klin.view(npages, page, hkv, d // 16, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        v = (
            vlin.view(npages, page // 16, 16, hkv, dv)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
        )
    elif layout == "linear3d":
        k, v = klin[:, 0], vlin[:, 0]
    else:
        k, v = klin, vlin
    cuq = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor(klens, dtype=torch.int32, device="cuda")
    output = torch.full(
        (sum(qlens), hq, dv), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    return SimpleNamespace(
        page=page,
        layout=layout,
        d=d,
        dv=dv,
        hq=hq,
        hkv=hkv,
        q=q,
        k=k,
        v=v,
        qs=qs,
        ks=ks,
        vs=vs,
        cuq=cuq,
        lengths=lengths,
        table=table,
        qlens=list(qlens),
        klens=list(klens),
        maxq=max(qlens, default=0),
        maxkv=maxkv,
        out=output,
    )


def reference(case, *, scale=None, lengths=None):
    """Logical token reconstruction and FP32 attention, without kernel helpers."""
    scale = case.d**-0.5 if scale is None else scale
    lengths = case.klens if lengths is None else lengths
    output = []
    offset = 0
    for b, (qlen, klen) in enumerate(zip(case.qlens, lengths)):
        if klen == 0:
            output.append(torch.zeros(qlen, case.hq, case.dv, device="cuda"))
            offset += qlen
            continue
        physical = case.table[b, : (klen + case.page - 1) // case.page].long()
        k, v = case.k[physical], case.v[physical]
        if case.layout == "vectorized":
            k, v = k.permute(0, 3, 1, 2, 4), v.permute(0, 2, 4, 1, 3)
        k = k.reshape(-1, case.hkv, case.d)[:klen].float() * case.ks
        v = v.reshape(-1, case.hkv, case.dv)[:klen].float() * case.vs
        q = case.q[offset : offset + qlen].float() * case.qs
        offset += qlen
        k = k.repeat_interleave(case.hq // case.hkv, dim=1)
        v = v.repeat_interleave(case.hq // case.hkv, dim=1)
        scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2) * scale
        allowed = (
            torch.arange(klen, device="cuda")[None, :]
            <= torch.arange(qlen, device="cuda")[:, None] + klen - qlen
        )
        probability = torch.softmax(
            scores.masked_fill(~allowed, float("-inf")), dim=-1
        ).nan_to_num(0)
        output.append((probability @ v.transpose(0, 1)).transpose(0, 1))
    return (
        torch.cat(output) if output else torch.empty_like(case.out, dtype=torch.float32)
    )


def run_case(case, **kwargs):
    options = {
        "block_table": case.table,
        "seqlen_k": case.lengths,
        "q_descale": case.qs,
        "k_descale": case.ks,
        "v_descale": case.vs,
        "out": case.out,
    }
    options.update(kwargs)
    return flydsl_flash_attn_paged_fp8_func(
        case.q, case.k, case.v, case.cuq, case.maxq, case.maxkv, **options
    )


def csr_metadata(case, prefix=0):
    counts = [(length + case.page - 1) // case.page for length in case.klens]
    offsets = [prefix]
    chunks = [torch.full((prefix,), -1, dtype=torch.int32, device=case.q.device)]
    for batch, count in enumerate(counts):
        offsets.append(offsets[-1] + count)
        chunks.append(case.table[batch, :count])
    chunks.append(torch.full((3,), -1, dtype=torch.int32, device=case.q.device))
    last = [(length - 1) % case.page + 1 if length else 0 for length in case.klens]
    return {
        "block_table": None,
        "kv_indptr": torch.tensor(offsets, dtype=torch.int32, device=case.q.device),
        "kv_page_indices": torch.cat(chunks),
        "kv_last_page_lens": torch.tensor(
            last, dtype=torch.int32, device=case.q.device
        ),
    }


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("lazy", [True, False])
def test_csr_ragged_and_odd_page_bases(page, layout, d, dv, lazy):
    case = make_case(
        page,
        layout,
        d,
        dv,
        qlens=(0, 65, 300, 17),
        klens=(33, 0, max(129, page + 1), 65),
    )
    options = csr_metadata(case, prefix=5)
    check_case(case, **options, dualwave_swp_lazy_rescale=lazy)


@gfx950
@pytest.mark.parametrize("layout", ["linear3d", "linear"])
@pytest.mark.parametrize("d,dv", DIMS)
def test_csr_page1_optional_last_lengths(layout, d, dv):
    case = make_case(1, layout, d, dv)
    options = csr_metadata(case)
    explicit = check_case(case, **options).clone()
    options["kv_last_page_lens"] = None
    torch.testing.assert_close(check_case(case, **options), explicit, rtol=0, atol=0)


def check_case(case, **kwargs):
    actual = run_case(case, **kwargs)
    torch.cuda.synchronize()
    assert actual is kwargs.get("out", case.out)
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(
        actual.float(),
        reference(case, scale=kwargs.get("softmax_scale")),
        rtol=0.02,
        atol=0.02,
    )
    return actual


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("lazy", [True, False])
def test_native_layouts_ragged_multiblock(page, layout, d, dv, lazy):
    check_case(make_case(page, layout, d, dv), dualwave_swp_lazy_rescale=lazy)


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_empty_requests_and_fully_masked_rows(page, layout, d, dv):
    case = make_case(
        page, layout, d, dv, qlens=(0, 65, 300, 17), klens=(33, 0, 129, 65)
    )
    actual = check_case(case)
    assert torch.count_nonzero(actual[:65]) == 0
    assert torch.count_nonzero(actual[65 : 65 + 171]) == 0


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_runtime_scale_reuses_compiled_launcher(page, layout, d, dv):
    case = make_case(page, layout, d, dv)
    default = check_case(case).clone()
    misses = paged._build.cache_info().misses
    for scale in (case.d**-0.5, 0.037, 0.137):
        actual = check_case(case, softmax_scale=scale)
        if scale == case.d**-0.5:
            torch.testing.assert_close(actual, default, rtol=0, atol=0)
    assert paged._build.cache_info().misses == misses


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS[:3])
@pytest.mark.parametrize("d,dv", DIMS)
def test_bounded_matches_wide_pointer_cache(monkeypatch, page, layout, d, dv):
    case = make_case(page, layout, d, dv)
    bounded = check_case(case).clone()
    build = paged._build

    def wide(**kwargs):
        assert kwargs["buffered"]
        kwargs["buffered"] = False
        return build(**kwargs)

    monkeypatch.setattr(paged, "_build", wide)
    torch.testing.assert_close(check_case(case), bounded, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [(1, "linear3d", 128, 128), (1, "linear", 192, 128), (16, "vectorized", 192, 192)],
)
def test_wide_cache_with_byte_offset_base(monkeypatch, page, layout, d, dv):
    case = make_case(page, layout, d, dv)

    def offset_copy(tensor):
        storage = torch.empty(
            tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device
        )
        view = storage[1:].view(tensor.shape)
        view.copy_(tensor)
        assert view.data_ptr() % 4 != 0
        return view

    case.k, case.v = offset_copy(case.k), offset_copy(case.v)
    build = paged._build

    def wide(**kwargs):
        kwargs["buffered"] = False
        return build(**kwargs)

    monkeypatch.setattr(paged, "_build", wide)
    check_case(case)


@gfx950
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("heads", [(16, 1), (8, 4)])
def test_scalar_and_paired_page64_paths(d, dv, heads):
    case = make_case(64, "vectorized", d, dv, heads=heads)
    even = check_case(case).clone()
    case.maxkv -= 64
    torch.testing.assert_close(check_case(case), even, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [
        (1, "linear3d", 192, 192),
        (16, "vectorized", 192, 128),
        (64, "vectorized", 128, 128),
        (1024, "vectorized", 192, 192),
    ],
)
def test_graph_reads_updated_lengths(page, layout, d, dv):
    case = make_case(page, layout, d, dv, qlens=(17,), klens=(max(257, page + 1),))
    run_case(case, softmax_scale=0.137)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_case(case, softmax_scale=0.137)
    assert captured is case.out
    for length in (65, 129, case.klens[0], 0, 33):
        case.lengths[0] = length
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(case.out.isfinite().all())
        torch.testing.assert_close(
            case.out.float(),
            reference(case, scale=0.137, lengths=[length]),
            rtol=0.02,
            atol=0.02,
        )


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
def test_strided_metadata_and_scalar_descales(page, layout):
    case = make_case(page, layout, 192, 128)
    case.cuq = case.cuq.repeat_interleave(2)[::2]
    case.lengths = case.lengths.repeat_interleave(2)[::2]
    case.table = case.table.repeat_interleave(2, dim=1)[:, ::2]
    expected = reference(case)
    actual = run_case(
        case,
        q_descale=case.qs.reshape(()),
        k_descale=case.ks.as_strided((1,), (0,)),
        v_descale=case.vs.reshape(1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [
        (1, "linear", 128, 128),
        (1, "linear3d", 192, 128),
        (16, "vectorized", 192, 192),
        (64, "vectorized", 192, 128),
        (1024, "vectorized", 192, 192),
    ],
)
def test_copies_follow_nondefault_stream(page, layout, d, dv):
    case = make_case(page, layout, d, dv)
    expected = check_case(case).clone()

    def strided(tensor):
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2),
            dtype=tensor.dtype,
            device="cuda",
        )
        view = storage[..., ::2]
        view.copy_(tensor)
        return view

    case.q, case.k, case.v = map(strided, (case.q, case.k, case.v))
    stream = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    run_case(case, stream=stream)
    stream.synchronize()
    torch.cuda._sleep(3_000_000_000)
    blocked = torch.cuda.Event()
    blocked.record()
    run_case(case, stream=stream)
    stream.synchronize()
    assert not blocked.query(), "copies waited on the blocked default stream"
    torch.cuda.synchronize()
    torch.testing.assert_close(case.out, expected, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_softmax_scale(scale):
    with pytest.raises(ValueError, match="positive and finite"):
        run_case(make_case(64, "vectorized", 128, 128), softmax_scale=scale)


@gfx950
@pytest.mark.parametrize("field", ["cuq", "lengths", "table"])
def test_rejects_metadata_dtype(field):
    case = make_case(64, "vectorized", 128, 128)
    setattr(case, field, getattr(case, field).to(torch.int64))
    with pytest.raises(ValueError, match="int32"):
        run_case(case)


@gfx950
@pytest.mark.parametrize("mode", ["query", "kv", "cache"])
def test_empty_attention_skips_builder(monkeypatch, mode):
    case = make_case(
        64,
        "vectorized",
        128,
        128,
        qlens=(0,) if mode == "query" else (17,),
        klens=(0,) if mode == "kv" else (65,),
    )
    if mode == "cache":
        case.k, case.v = case.k[:0], case.v[:0]

    def unexpected(**kwargs):
        raise AssertionError("empty attention reached the kernel builder")

    monkeypatch.setattr(paged, "_build", unexpected)
    actual = run_case(case)
    torch.cuda.synchronize()
    assert actual is case.out
    assert torch.count_nonzero(actual) == 0


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("rectangular", [False, True])
def test_public_batch_prefill_routes_to_flydsl(
    monkeypatch, page, layout, d, dv, rectangular
):
    from aiter.ops import mha

    case = make_case(page, layout, d, dv)
    metadata = csr_metadata(case, prefix=5)
    indptr, indices = metadata.pop("kv_indptr"), metadata.pop("kv_page_indices")
    if rectangular:
        metadata["block_table"] = case.table
        indices.fill_(-1)  # The rectangular table must take precedence.
    seen = []
    direct = paged.flydsl_flash_attn_paged_fp8_func

    def observed(*args, **kwargs):
        seen.append(kwargs)
        return direct(*args, **kwargs)

    def unexpected_ck(*args, **kwargs):
        raise AssertionError("supported paged FP8 request reached CK")

    monkeypatch.setattr(paged, "flydsl_flash_attn_paged_fp8_func", observed)
    monkeypatch.setattr(mha, "_mha_batch_prefill", unexpected_ck)
    actual = mha.mha_batch_prefill_func(
        case.q,
        case.k,
        case.v,
        case.cuq,
        indptr,
        indices,
        case.maxq,
        case.maxkv,
        causal=True,
        softmax_scale=0.137,
        out=case.out,
        seqlen_k=case.lengths,
        q_descale=case.qs,
        k_descale=case.ks,
        v_descale=case.vs,
        **metadata,
    )
    torch.cuda.synchronize()
    assert actual is case.out and len(seen) == 1
    torch.testing.assert_close(
        actual.float(), reference(case, scale=0.137), rtol=0.02, atol=0.02
    )


@gfx950
@pytest.mark.parametrize(
    "feature",
    [
        "noncausal",
        "dropout",
        "softcap",
        "window",
        "alibi",
        "sink",
        "sink_size",
        "lse",
        "probabilities",
        "block_scale",
        "invalid_scale",
    ],
)
def test_public_unsupported_options_fall_through(monkeypatch, feature):
    from aiter.ops import mha

    case = make_case(16, "vectorized", 128, 128, qlens=(17,), klens=(65,))
    metadata = csr_metadata(case)
    indptr, indices = metadata.pop("kv_indptr"), metadata.pop("kv_page_indices")
    options = {
        "causal": True,
        "out": case.out,
        "q_descale": case.qs,
        "k_descale": case.ks,
        "v_descale": case.vs,
    }
    changes = {
        "noncausal": {"causal": False},
        "dropout": {"dropout_p": 0.1},
        "softcap": {"logits_soft_cap": 1.0},
        "window": {"window_size": (128, 0)},
        "alibi": {"alibi_slopes": torch.ones(case.hq, device="cuda")},
        "sink": {"sink_ptr": torch.ones(case.hq, device="cuda")},
        "sink_size": {"sink_size": 1},
        "lse": {"return_lse": True},
        "probabilities": {"return_attn_probs": True},
        "block_scale": {
            "kv_block_descale": torch.ones(case.k.shape[0], case.hkv, 2, device="cuda"),
            "k_descale": None,
            "v_descale": None,
        },
        "invalid_scale": {"softmax_scale": -1.0},
    }
    options.update(changes[feature])
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            case.out.zero_(),
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            None,
        )

    def unexpected_direct(*args, **kwargs):
        raise AssertionError("unsupported feature entered paged FlyDSL")

    monkeypatch.setattr(mha, "_mha_batch_prefill", fallback)
    monkeypatch.setattr(paged, "flydsl_flash_attn_paged_fp8_func", unexpected_direct)
    mha.mha_batch_prefill_func(
        case.q,
        case.k,
        case.v,
        case.cuq,
        indptr,
        indices,
        case.maxq,
        case.maxkv,
        **metadata,
        **options,
    )
    assert len(calls) == 1
    assert calls[0][1]["causal"] == options["causal"]
    assert calls[0][1]["q_descale"] is case.qs


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [
        (1, "linear3d", 192, 128),
        (16, "vectorized", 192, 192),
        (64, "vectorized", 128, 128),
        (1024, "vectorized", 192, 128),
    ],
)
def test_csr_graph_updates_ranges_and_lengths(page, layout, d, dv):
    case = make_case(
        page, layout, d, dv, qlens=(17, 65), klens=(max(513, page + 1), 81)
    )
    metadata = csr_metadata(case, prefix=5)
    run_case(case, **metadata, softmax_scale=0.137)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_case(case, **metadata, softmax_scale=0.137)
    for first_length in (65, 129, case.klens[0], 0, 33):
        lengths = [first_length, 81]
        counts = [(n + page - 1) // page for n in lengths]
        offsets = [5, 5 + counts[0], 5 + sum(counts)]
        metadata["kv_indptr"].copy_(
            torch.tensor(offsets, dtype=torch.int32, device="cuda")
        )
        metadata["kv_last_page_lens"].copy_(
            torch.tensor(
                [(n - 1) % page + 1 if n else 0 for n in lengths],
                dtype=torch.int32,
                device="cuda",
            )
        )
        metadata["kv_page_indices"].fill_(-1)
        for batch, count in enumerate(counts):
            metadata["kv_page_indices"][offsets[batch] : offsets[batch + 1]].copy_(
                case.table[batch, :count]
            )
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(case.out.isfinite().all())
        torch.testing.assert_close(
            case.out.float(),
            reference(case, scale=0.137, lengths=lengths),
            rtol=0.02,
            atol=0.02,
        )


@gfx950
@pytest.mark.parametrize("page_base", [0, 1, 5])
@pytest.mark.parametrize("num_pages,tile", [(0, 0), (1, 0), (3, 2), (3, 4), (4, 2)])
def test_csr_paired_ids_are_bounded_and_byte_exact(page_base, num_pages, tile):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_memory import store
    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
        DualwaveFp8KernelContext,
    )

    @flyc.kernel
    def lookup(table: fx.Tensor, output: fx.Tensor):
        ctx = SimpleNamespace(
            traits=SimpleNamespace(
                PAGE_SIZE=64, BLOCK_N=64, PAIRED_PAGE_IDS=True, CSR_PAGE_TABLE=True
            ),
            BlockTable=table,
            page_base=fx.Int64(page_base),
            request_page_count=fx.Int64(num_pages),
            seqlen_kv_v=fx.Int64(num_pages * 64),
        )
        DualwaveFp8KernelContext.init_page_table(ctx)
        first, second = DualwaveFp8KernelContext.load_page_id_pair(
            ctx, fx.Int64(tile * 64)
        )
        values = fx.Vector.from_elements([fx.Int32(first), fx.Int32(second)], fx.Int32)
        store(fx.add_offset(fx.get_iter(output), fx.Int64(fx.thread_idx.x) * 2), values)

    @flyc.jit
    def launch(table: fx.Tensor, output: fx.Tensor):
        lookup(table, output).launch(grid=(1, 1, 1), block=(64, 1, 1))

    table = torch.full((page_base + 5,), -1, dtype=torch.int32, device="cuda")
    table[page_base : page_base + num_pages] = torch.arange(
        11, 11 + num_pages, dtype=torch.int32, device="cuda"
    )
    output = torch.empty((64, 2), dtype=torch.int32, device="cuda")
    launch(table, output.flatten())
    torch.cuda.synchronize()
    expected = torch.tensor(
        [11 + n if n < num_pages else 0 for n in (tile, tile + 1)],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(output, expected.expand_as(output), rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
def test_empty_csr_with_nonzero_launch_bound(monkeypatch, page, layout):
    case = make_case(page, layout, 192, 128, qlens=(17,), klens=(0,))
    case.maxkv = 1024
    metadata = csr_metadata(case)
    metadata["kv_page_indices"] = torch.empty(0, dtype=torch.int32, device="cuda")

    def unexpected(**kwargs):
        raise AssertionError("empty CSR reached the attention builder")

    monkeypatch.setattr(paged, "_build", unexpected)
    output = run_case(case, **metadata)
    torch.cuda.synchronize()
    assert output is case.out and torch.count_nonzero(output) == 0


@gfx950
@pytest.mark.parametrize("page,layout", [(1, "linear3d"), (64, "vectorized")])
def test_csr_strided_metadata_on_side_stream(page, layout):
    case = make_case(page, layout, 192, 128)
    metadata = csr_metadata(case, prefix=5)
    for name in ("kv_indptr", "kv_page_indices", "kv_last_page_lens"):
        metadata[name] = metadata[name].repeat_interleave(2)[::2]
    stream = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    run_case(case, **metadata, stream=stream)
    stream.synchronize()
    torch.cuda._sleep(3_000_000_000)
    blocked = torch.cuda.Event()
    blocked.record()
    run_case(case, **metadata, stream=stream)
    stream.synchronize()
    assert not blocked.query(), "CSR normalization waited on the default stream"
    torch.cuda.synchronize()
    torch.testing.assert_close(case.out.float(), reference(case), rtol=0.02, atol=0.02)


@gfx950
@pytest.mark.parametrize(
    "violation", ["missing_last", "last_dtype", "indptr_shape", "indices_dtype"]
)
def test_csr_rejects_invalid_metadata(violation):
    case = make_case(16, "vectorized", 128, 128)
    metadata = csr_metadata(case)
    if violation == "missing_last":
        metadata["kv_last_page_lens"] = None
    elif violation == "last_dtype":
        metadata["kv_last_page_lens"] = metadata["kv_last_page_lens"].to(torch.int64)
    elif violation == "indptr_shape":
        metadata["kv_indptr"] = metadata["kv_indptr"][:-1]
    else:
        metadata["kv_page_indices"] = metadata["kv_page_indices"].to(torch.int64)
    with pytest.raises(ValueError, match="kv_"):
        run_case(case, **metadata)
